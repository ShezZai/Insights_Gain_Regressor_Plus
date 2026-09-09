#!/usr/bin/env python3
"""Add intraday price features to public.articles, per article + primary_ticker.

Three columns, all computed from 1-minute bars (Massive aggs API):

  vix_at_article_time              VIXY price at the article's minute. VIXY (the
                                   short-term VIX futures ETF) stands in for the
                                   VIX index itself, which Massive does not serve
                                   -- `I:VIX` errors, `VIX`/`^VIX` come back empty.
  aiq_ai_etf_intraday_gain_till_article AIQ (Global X Artificial Intelligence &
                                   Technology) price at article time vs its own
                                   09:30 open -- the sector move to control for.
                                   AIQ prints in ~385 of 390 RTH minutes; CHAT
                                   (109/390) is too thin to time an article with.
  intraday_gain_till_article       primary_ticker: price at article time vs the
                                   09:30 ET regular-session open, as a fraction.
  gain_90m_after_article           primary_ticker: price 90 minutes after the
                                   article vs price at article time, as a fraction.
  intraday_gain_till_article_first the same two, measured on
  gain_90m_after_article_first     first_mentioned_ticker instead -- the name the
                                   article actually talks about first, rather
                                   than whichever ticker the feed listed first.
                                   Pick with --ticker-column first, or
                                   first_or_listed to fall back to
                                   first_listed_ticker where the text names no
                                   universe ticker.

Note the SQL-legal name `gain_90m_after_article` -- an identifier cannot start
with a digit, so `90_mins_gain_after_article` would need quoting everywhere.

Assumes the corpus is already filtered to NYSE trading days between the open
and 90 minutes before the close (see the news_trading_window DB), so the
+90-minute mark always lands inside the regular session.

Work splits into two phases, so each can run (and resume) alone:
  index   whole-market series -- VIXY and AIQ -- which every article shares.
          19 calls per series covers the corpus, regardless of article count.
  ticker  the primary_ticker bars, which is the expensive phase.

Rate limit is the whole game: the Massive key allows ~5 requests/minute, so
bars are fetched one (ticker, calendar-month) at a time -- a month of 1-minute
bars fits in a single 50k-row page -- instead of one call per ticker-day.
That turns ~3,600 calls (12h) into ~950 (3h). Each month's rows are written
before moving on, so an interrupted run resumes where it stopped: rows that
already have all three values are skipped unless --reprocess.

Usage:
    python add_price_features.py --limit 50        # smoke test
    python add_price_features.py                   # whole table
    python add_price_features.py --rpm 5           # requests/minute budget
    python add_price_features.py --phase index --series ai_etf
    python add_price_features.py --reprocess --ids 78,4085

Requires MASSIVE_API_KEY (from .env via ticker_news.shared.config).
"""

from __future__ import annotations

import argparse
import bisect
import time
from collections import defaultdict
from datetime import datetime, date, time as dtime, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

import psycopg
import requests

from ticker_news.research.market_data import AGGS_URL, api_key

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = dtime(9, 30)
REGULAR_CLOSE = dtime(16, 0)
VIX_PROXY = "VIXY"
AI_ETF = "AIQ"
AFTER_MINUTES = 90
PAGE_LIMIT = 50_000
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

DDL = """
ALTER TABLE public.articles
    ADD COLUMN IF NOT EXISTS vix_at_article_time                   double precision,
    ADD COLUMN IF NOT EXISTS aiq_ai_etf_intraday_gain_till_article double precision,
    ADD COLUMN IF NOT EXISTS intraday_gain_till_article            double precision,
    ADD COLUMN IF NOT EXISTS gain_90m_after_article                double precision,
    ADD COLUMN IF NOT EXISTS intraday_gain_till_article_first      double precision,
    ADD COLUMN IF NOT EXISTS gain_90m_after_article_first          double precision
"""


class RateLimiter:
    """Global minimum spacing between API requests (the key allows ~5/min)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._last = 0.0

    def wait(self) -> None:
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


class DayBars:
    """Regular-session 1-minute bars for one ticker-day, indexed by ET minute."""

    def __init__(self, rows: list[tuple[datetime, float, float]]):
        rows.sort(key=lambda r: r[0])
        self.times = [r[0] for r in rows]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def session_open(self) -> float | None:
        """Open of the first regular-session bar (the 09:30 print)."""
        return self.rows[0][1] if self.rows else None

    def price_at(self, ts: datetime) -> float | None:
        """Close of the bar covering `ts`, else the last bar before it."""
        pos = bisect.bisect_right(self.times, ts) - 1
        return self.rows[pos][2] if pos >= 0 else None


def month_bounds(ym: str) -> tuple[str, str]:
    first = date.fromisoformat(f"{ym}-01")
    nxt = date(first.year + (first.month == 12), (first.month % 12) + 1, 1)
    return first.isoformat(), (nxt - timedelta(days=1)).isoformat()


def fetch_month(ticker: str, ym: str, key: str, limiter: RateLimiter) -> dict[date, DayBars]:
    """Every regular-session minute bar for `ticker` in month `ym`, by ET date."""
    frm, to = month_bounds(ym)
    url = AGGS_URL.format(ticker=ticker, multiplier=1, span="minute", frm=frm, to=to)
    params: dict = {"adjusted": "true", "sort": "asc", "limit": PAGE_LIMIT, "apiKey": key}
    per_day: dict[date, list] = defaultdict(list)

    while url:
        payload = None
        for attempt in range(MAX_RETRIES):
            limiter.wait()
            try:
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"transient {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"{ticker} {ym}: {exc!r}") from exc
                time.sleep(5.0 * (attempt + 1))
        for bar in (payload.get("results") or []):
            ts = datetime.fromtimestamp(bar["t"] / 1000, tz=ET)
            if REGULAR_OPEN <= ts.time() < REGULAR_CLOSE:
                per_day[ts.date()].append((ts, float(bar["o"]), float(bar["c"])))
        url = payload.get("next_url")
        params = {"apiKey": key}

    return {day: DayBars(rows) for day, rows in per_day.items()}


PHASE_COLUMNS = {
    "index": ("vix_at_article_time", "aiq_ai_etf_intraday_gain_till_article"),
}
# --ticker-column: which ticker the gains are measured on, and where they land.
TICKER_COLUMNS = {
    "primary": ("primary_ticker",
                ("intraday_gain_till_article", "gain_90m_after_article")),
    "first": ("first_mentioned_ticker",
              ("intraday_gain_till_article_first", "gain_90m_after_article_first")),
    # Same target columns, but falls back to the feed's first universe tag for
    # articles that name no universe ticker in their text (or aren't scraped).
    "first_or_listed": ("COALESCE(first_mentioned_ticker, first_listed_ticker)",
                        ("intraday_gain_till_article_first", "gain_90m_after_article_first")),
}
SERIES_TICKER = {"vix": VIX_PROXY, "ai_etf": AI_ETF}


def load_articles(conn, columns, ids, limit, reprocess, ticker_column=None):
    """Articles still missing any of `columns` (all of them with --reprocess).

    `ticker_column` is the column the bars come from; the index phase passes
    None because VIXY/AIQ are the same for every article, ticker or not.
    """
    where = ["published_utc IS NOT NULL"]
    if ticker_column:
        where.append(f"{ticker_column} IS NOT NULL")
    params: list = []
    if ids:
        where.append("id = ANY(%s)")
        params.append(ids)
    elif not reprocess:
        where.append("(" + " OR ".join(f"{c} IS NULL" for c in columns) + ")")
    selected = ticker_column or "NULL::text"
    sql = (f"SELECT id, {selected}, published_utc FROM public.articles "
           f"WHERE {' AND '.join(where)} ORDER BY published_utc")
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def write(conn, columns: Sequence[str], rows: list[tuple]) -> None:
    """UPDATE `columns` (values first, id last in each row tuple)."""
    if not rows:
        return
    sets = ", ".join(f"{c} = %s" for c in columns)
    with conn.cursor() as cur:
        cur.executemany(f"UPDATE public.articles SET {sets} WHERE id = %s", rows)


def months_of(articles) -> list[str]:
    return sorted({p.astimezone(ET).strftime("%Y-%m") for _, _, p in articles})


def index_phase(conn, articles, series: Sequence[str], key, limiter, stats) -> None:
    """Fill the whole-market columns: VIXY price and the AI-ETF intraday gain.

    Both series are the same for every article, so this costs one call per
    (series, month) no matter how many articles there are.
    """
    months = months_of(articles)
    bars: dict[str, dict[date, DayBars]] = {}
    for name in series:
        symbol = SERIES_TICKER[name]
        got: dict[date, DayBars] = {}
        for ym in months:
            try:
                got.update(fetch_month(symbol, ym, key, limiter))
            except Exception as exc:
                stats["errors"] += 1
                print(f"[{symbol} error] {exc!r}", flush=True)
        bars[name] = got
        print(f"[{name}] {symbol}: {len(got)} day(s) of bars", flush=True)

    columns = [PHASE_COLUMNS["index"][0] if n == "vix" else PHASE_COLUMNS["index"][1]
               for n in series]
    rows: list[tuple] = []
    for article_id, _ticker, published in articles:
        et = published.astimezone(ET)
        values: list = []
        for name in series:
            day = bars[name].get(et.date())
            price = day.price_at(et) if day else None
            if name == "vix":
                value = price
            else:  # AI-ETF gain from its own 09:30 open
                open_px = day.session_open() if day else None
                value = (price / open_px - 1.0) if (price and open_px) else None
            stats[name] += value is not None
            values.append(value)
        rows.append(tuple(values) + (article_id,))
    write(conn, columns, rows)
    print(f"[index] wrote {len(rows)} row(s) :: {stats}", flush=True)


def ticker_phase(conn, articles, columns, key, limiter, rpm, stats) -> None:
    """Fill a pair of gain columns -- one call per (ticker, month)."""
    by_month: dict[tuple[str, str], list[tuple[int, datetime]]] = defaultdict(list)
    for article_id, ticker, published in articles:
        et = published.astimezone(ET)
        by_month[(ticker.upper(), et.strftime("%Y-%m"))].append((article_id, et))
    print(f"[ticker] {len(by_month)} ticker-month call(s)"
          f" ~= {len(by_month) / rpm / 60:.1f}h at {rpm}/min", flush=True)

    started = time.monotonic()
    for done, (ticker, ym) in enumerate(sorted(by_month), start=1):
        try:
            days = fetch_month(ticker, ym, key, limiter)
        except Exception as exc:
            stats["errors"] += 1
            print(f"[bars error] {exc!r}", flush=True)
            continue

        rows: list[tuple] = []
        for article_id, et in by_month[(ticker, ym)]:
            bars = days.get(et.date())
            if bars is None or not len(bars):
                stats["no_bars"] += 1
                continue
            open_px = bars.session_open()
            at = bars.price_at(et)
            after = bars.price_at(et + timedelta(minutes=AFTER_MINUTES))
            gain_till = (at / open_px - 1.0) if (at and open_px) else None
            gain_after = (after / at - 1.0) if (at and after) else None
            stats["gain_till"] += gain_till is not None
            stats["gain_after"] += gain_after is not None
            rows.append((gain_till, gain_after, article_id))
        write(conn, columns, rows)

        if done % 10 == 0 or done == len(by_month):
            rate = (time.monotonic() - started) / done
            print(f"[progress] {done}/{len(by_month)} ticker-months :: {stats} "
                  f":: eta {rate * (len(by_month) - done) / 60:.0f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="dbname=news_trading_window", help="Postgres DSN")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N articles")
    ap.add_argument("--ids", default=None, help="Comma-separated article ids")
    ap.add_argument("--reprocess", action="store_true", help="Recompute rows that already have values")
    ap.add_argument("--rpm", type=float, default=4.8, help="Request budget per minute (key allows ~5)")
    ap.add_argument("--phase", choices=["index", "ticker", "both"], default="both",
                    help="index = VIXY/AI-ETF columns; ticker = the per-name gains")
    ap.add_argument("--ticker-column", choices=sorted(TICKER_COLUMNS), default="primary",
                    help="ticker phase: measure gains on primary_ticker or first_mentioned_ticker")
    ap.add_argument("--series", default="vix,ai_etf",
                    help=f"index phase only: which of {sorted(SERIES_TICKER)} to fetch")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    series = [s.strip() for s in args.series.split(",") if s.strip()]
    unknown = set(series) - set(SERIES_TICKER)
    if unknown:
        ap.error(f"unknown --series {sorted(unknown)}; choose from {sorted(SERIES_TICKER)}")

    key = api_key()
    limiter = RateLimiter(args.rpm)
    stats = {"vix": 0, "ai_etf": 0, "gain_till": 0, "gain_after": 0,
             "no_bars": 0, "errors": 0}

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(DDL)

        if args.phase in ("index", "both"):
            columns = [PHASE_COLUMNS["index"][0] if n == "vix" else PHASE_COLUMNS["index"][1]
                       for n in series]
            articles = load_articles(conn, columns, ids, args.limit, args.reprocess)
            print(f"[index] {len(articles)} article(s) for {series}", flush=True)
            if articles:
                index_phase(conn, articles, series, key, limiter, stats)

        if args.phase in ("ticker", "both"):
            ticker_column, columns = TICKER_COLUMNS[args.ticker_column]
            articles = load_articles(conn, columns, ids, args.limit, args.reprocess,
                                     ticker_column=ticker_column)
            print(f"[ticker] {len(articles)} article(s) on {ticker_column}", flush=True)
            if articles:
                ticker_phase(conn, articles, columns, key, limiter, args.rpm, stats)

    print(f"[done] {stats}", flush=True)


if __name__ == "__main__":
    main()
