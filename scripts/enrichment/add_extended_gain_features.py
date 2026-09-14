#!/usr/bin/env python3
"""Add longer-horizon price-gain features to public.articles, per article + primary_ticker.

Two columns, computed from Yahoo Finance bars (yfinance -- free, no API key):

  gain_24h_after_article  primary_ticker: price at the first available trade
                          at/after (article time + 24h), vs the price at/before
                          article time, as a fraction. Hourly bars, extended
                          hours included (prepost=True).
  gain_7d_after_article   Same, but +7 days and daily bars.

This mirrors add_price_features.py's gain_90m_after_article, but on longer
horizons where minute-bar precision isn't needed and crossing into a new
session/day/weekend is the point rather than something to avoid:

  - The *anchor* price (at article time) is the last trade at/before it --
    same "price known at publish time" semantics as add_price_features.py's
    DayBars.price_at().
  - The *target* price (at article time + 24h / + 7d) is the first trade
    AT/AFTER it -- the first price actually tradeable at or beyond that
    horizon. This is a deliberate asymmetry from the anchor side, and it
    means weekends/holidays need no special handling: a Friday article's
    target simply bisects forward to the next available Monday bar.

Data source is per-ticker, not per-(ticker, month) like Massive: one
yfinance call fetches a ticker's whole history over that ticker's own
article date range, so the total call count is roughly the number of
distinct primary_ticker values (~100-120), not one per ticker-month. No
API key or formal rate limit; a small per-ticker delay guards against
soft IP-throttling.

yfinance rejects an hourly request outright if its span exceeds ~730
days (not just the portion beyond that), so the hourly fetch's start
date is clipped to that rolling floor -- articles older than ~730 days
lose hourly (+24h) coverage but keep daily (+7d), which has no such
limit.

Row eligibility: scripts/enrichment/build_nyse_trading_window.sql already
restricts the corpus to articles with 90 minutes of same-day headroom
before market close -- a strict superset of what +24h/+7d need, so no
additional filtering is required here.

Usage:
    python add_extended_gain_features.py --limit 20     # smoke test
    python add_extended_gain_features.py                # whole table
    python add_extended_gain_features.py --reprocess --ids 78,4085
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg
import yfinance as yf

MARKET_TZ = ZoneInfo("America/New_York")
AFTER_HOURS = 24
AFTER_DAYS = 7
TICKER_DELAY = 0.75  # seconds between tickers, soft insurance against throttling
HOURLY_LOOKBACK_DAYS = 729  # yfinance rejects hourly requests older than ~730 days

DDL = """
ALTER TABLE public.articles
    ADD COLUMN IF NOT EXISTS gain_24h_after_article double precision,
    ADD COLUMN IF NOT EXISTS gain_7d_after_article  double precision
"""

COLUMNS = ("gain_24h_after_article", "gain_7d_after_article")


def price_at_or_before(index: pd.DatetimeIndex, closes: np.ndarray, ts: pd.Timestamp) -> float | None:
    """Close of the last bar at/before `ts` -- the anchor price at article time."""
    pos = index.searchsorted(ts, side="right") - 1
    return float(closes[pos]) if pos >= 0 else None


def price_at_or_after(index: pd.DatetimeIndex, closes: np.ndarray, ts: pd.Timestamp) -> float | None:
    """Close of the first bar at/after `ts` -- the horizon target price."""
    pos = index.searchsorted(ts, side="left")
    return float(closes[pos]) if pos < len(index) else None


def fetch_with_retry(ticker: str, interval: str, start, end, max_retries: int = 3,
                     backoff: float = 5.0) -> pd.DataFrame | None:
    for attempt in range(max_retries):
        try:
            df = yf.Ticker(ticker).history(
                interval=interval, start=start, end=end,
                prepost=(interval != "1d"), auto_adjust=True,
            )
            if df is not None and not df.empty:
                return df
            return df  # empty but no error -- likely a bad/unresolved ticker, not transient
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"[yfinance error] {ticker} {interval}: {exc!r}", flush=True)
                return None
            time.sleep(backoff * (attempt + 1))
    return None


def load_articles(conn, ids, limit, reprocess):
    """Articles still missing either new column (both with --reprocess)."""
    where = ["published_utc IS NOT NULL", "primary_ticker IS NOT NULL"]
    params: list = []
    if ids:
        where.append("id = ANY(%s)")
        params.append(ids)
    elif not reprocess:
        where.append("(" + " OR ".join(f"{c} IS NULL" for c in COLUMNS) + ")")
    sql = (f"SELECT id, primary_ticker, published_utc FROM public.articles "
           f"WHERE {' AND '.join(where)} ORDER BY published_utc")
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def write(conn, rows: list[tuple]) -> None:
    """UPDATE COLUMNS (values first, id last in each row tuple)."""
    if not rows:
        return
    sets = ", ".join(f"{c} = %s" for c in COLUMNS)
    with conn.cursor() as cur:
        cur.executemany(f"UPDATE public.articles SET {sets} WHERE id = %s", rows)


def ticker_phase(conn, articles, stats) -> None:
    by_ticker: dict[str, list[tuple[int, pd.Timestamp]]] = {}
    for article_id, ticker, published in articles:
        et = pd.Timestamp(published).tz_convert(MARKET_TZ)
        by_ticker.setdefault(ticker.upper(), []).append((article_id, et))

    # Per-ticker range, not a corpus-wide one: a global range spanning the
    # whole corpus (often >730 days) makes yfinance reject the *entire*
    # hourly request, not just the old portion of it.
    hourly_floor = date.today() - timedelta(days=HOURLY_LOOKBACK_DAYS)

    print(f"[ticker] {len(by_ticker)} ticker(s)", flush=True)

    started = time.monotonic()
    for done, ticker in enumerate(sorted(by_ticker), start=1):
        ets = [et for _, et in by_ticker[ticker]]
        start = min(ets).date() - timedelta(days=1)
        end = (max(ets) + timedelta(days=AFTER_DAYS + 3)).date()

        hourly_start = max(start, hourly_floor)
        hourly = (fetch_with_retry(ticker, "1h", hourly_start, end)
                  if hourly_start < end else None)
        daily = fetch_with_retry(ticker, "1d", start, end)
        time.sleep(TICKER_DELAY)

        # Hourly and daily are independent sources -- e.g. yfinance's ~730-day
        # rolling window can empty out hourly for old articles while daily
        # (effectively unlimited history) still works, and vice versa isn't
        # impossible either. A gap in one must not suppress the other.
        h_idx = h_close = None
        if hourly is not None and not hourly.empty:
            h_idx, h_close = hourly.index, hourly["Close"].to_numpy()
        else:
            stats["hourly_empty"] += 1

        d_idx = d_close = None
        if daily is not None and not daily.empty:
            d_idx, d_close = daily.index, daily["Close"].to_numpy()
        else:
            stats["daily_empty"] += 1

        if hourly is None and daily is None:
            stats["errors"] += 1
        if h_idx is None and d_idx is None:
            continue

        rows: list[tuple] = []
        for article_id, et in by_ticker[ticker]:
            gain_24h = None
            if h_idx is not None:
                anchor_24h = price_at_or_before(h_idx, h_close, et)
                target_24h = price_at_or_after(h_idx, h_close, et + timedelta(hours=AFTER_HOURS))
                gain_24h = (target_24h / anchor_24h - 1.0) if (anchor_24h and target_24h) else None

            gain_7d = None
            if d_idx is not None:
                anchor_7d = price_at_or_before(d_idx, d_close, et)
                target_7d = price_at_or_after(d_idx, d_close, et + timedelta(days=AFTER_DAYS))
                gain_7d = (target_7d / anchor_7d - 1.0) if (anchor_7d and target_7d) else None

            stats["gain_24h"] += gain_24h is not None
            stats["gain_7d"] += gain_7d is not None
            rows.append((gain_24h, gain_7d, article_id))
        write(conn, rows)

        if done % 10 == 0 or done == len(by_ticker):
            rate = (time.monotonic() - started) / done
            print(f"[progress] {done}/{len(by_ticker)} ticker(s) :: {stats} "
                  f":: eta {rate * (len(by_ticker) - done) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="dbname=news_trading_window", help="Postgres DSN")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N articles")
    ap.add_argument("--ids", default=None, help="Comma-separated article ids")
    ap.add_argument("--reprocess", action="store_true", help="Recompute rows that already have values")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    stats = {"gain_24h": 0, "gain_7d": 0, "hourly_empty": 0, "daily_empty": 0, "errors": 0}

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(DDL)

        articles = load_articles(conn, ids, args.limit, args.reprocess)
        print(f"[ticker] {len(articles)} article(s)", flush=True)
        if articles:
            ticker_phase(conn, articles, stats)

    print(f"[done] {stats}", flush=True)


if __name__ == "__main__":
    main()
