#!/usr/bin/env python3
"""Add article metadata rows from the Massive news endpoint (no scraping).

Extends the corpus in time without changing what it is: rows land only if the
article was published on an official NYSE trading day between the regular open
and 90 minutes before the close -- the same filter that defines this database.

Rows are inserted with `status = 'pending'`, content NULL. The scraper's
`exists_ok` only skips `status = 'ok'`, so a later
`ticker-news scrape --csv ...` (or a re-run of the CSV export) fills them in.
An article returned for several tickers is stored once with the ticker lists
merged; existing scraped rows are never modified.

Tickers default to every distinct `primary_ticker` already in the table.

Rate limit: the Massive key allows ~5 requests/minute, shared with the aggs
endpoint, so calls are paced globally and pagination pages count too.

Usage:
    python fetch_more_articles.py --dry-run
    python fetch_more_articles.py
    python fetch_more_articles.py --tickers NVDA,AMD --range 2026-06-01:2026-09-09
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg
import requests

from ticker_news.ingestion.massive_rest import BASE_URL, PAGE_LIMIT
from ticker_news.scraping.urls import canonicalize_url, domain_of
from ticker_news.shared.config import get_settings
from ticker_news.shared.db import resolve_dsn

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = dtime(9, 30)
CLOSE_BUFFER_MIN = 90
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

# NYSE full closures spanning the fetchable range (2024-09-10 .. 2026-09-09).
HOLIDAYS = {
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09",  # 2025-01-09: national day of mourning (Carter)
    "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day observed (Jul 4 falls on a Saturday)
    "2026-09-07",  # Labor Day
}
# Early closes (13:00 ET), so the cutoff that day is 11:30 rather than 14:30.
EARLY_CLOSE = {
    "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
}

INSERT = """
INSERT INTO public.articles
    (url, url_canonical, source_domain, publisher, tickers, published_utc,
     title, author, status)
VALUES (%(url)s, %(url_canonical)s, %(source_domain)s, %(publisher)s, %(tickers)s,
        %(published_utc)s, %(title)s, %(author)s, 'pending')
ON CONFLICT (url) DO UPDATE
    SET tickers = ARRAY(SELECT DISTINCT unnest(public.articles.tickers || EXCLUDED.tickers))
    WHERE public.articles.status = 'pending'
"""


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._last = 0.0

    def wait(self) -> None:
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


def in_window(published: datetime) -> bool:
    """True if the article lands in the open .. close-90min NYSE window."""
    et = published.astimezone(ET)
    day = et.date().isoformat()
    if et.weekday() > 4 or day in HOLIDAYS:
        return False
    close = dtime(13, 0) if day in EARLY_CLOSE else dtime(16, 0)
    cutoff = (datetime.combine(date.today(), close) - timedelta(minutes=CLOSE_BUFFER_MIN)).time()
    return REGULAR_OPEN <= et.time() < cutoff


def fetch(ticker: str, start: str, end: str, key: str, limiter: RateLimiter) -> list[dict]:
    """Every article for `ticker` in [start, end], following cursor pages."""
    url = BASE_URL
    params: dict | None = {
        "ticker": ticker,
        "published_utc.gte": f"{start}T00:00:00Z",
        "published_utc.lte": f"{end}T23:59:59Z",
        "order": "asc", "sort": "published_utc",
        "limit": PAGE_LIMIT, "apiKey": key,
    }
    out: list[dict] = []
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
                    raise RuntimeError(f"{ticker} {start}..{end}: {exc!r}") from exc
                time.sleep(15.0)
        out.extend(payload.get("results") or [])
        url = payload.get("next_url")
        params = {"apiKey": key} if url else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default: DATABASE_URL / NEWS_DB_DSN from .env)")
    ap.add_argument("--tickers", default=None, help="Comma-separated; default = every primary_ticker in the table")
    ap.add_argument("--range", dest="ranges", action="append", default=None,
                    help="START:END (repeatable). Default: the two gap ranges around the corpus.")
    ap.add_argument("--rpm", type=float, default=4.8)
    ap.add_argument("--dry-run", action="store_true", help="Fetch and report, write nothing")
    args = ap.parse_args()

    today = datetime.now(tz=ET).date().isoformat()
    ranges = [tuple(r.split(":", 1)) for r in (args.ranges or
              [f"2024-09-10:2024-10-30", f"2026-06-01:{today}"])]
    key = get_settings().massive_api_key
    limiter = RateLimiter(args.rpm)

    with psycopg.connect(resolve_dsn(args.dsn), autocommit=True) as conn:
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT primary_ticker FROM public.articles "
                            "WHERE primary_ticker IS NOT NULL ORDER BY 1")
                tickers = [r[0] for r in cur.fetchall()]

        calls = len(tickers) * len(ranges)
        print(f"[plan] {len(tickers)} ticker(s) x {len(ranges)} range(s) = {calls}+ call(s)"
              f" ~= {calls / args.rpm:.0f} min at {args.rpm}/min :: {ranges}", flush=True)

        seen: dict[str, dict] = {}
        stats = {"returned": 0, "in_window": 0, "inserted": 0, "errors": 0}
        started = time.monotonic()

        for done, ticker in enumerate(tickers, start=1):
            rows: dict[str, dict] = {}
            for start, end in ranges:
                try:
                    articles = fetch(ticker, start, end, key, limiter)
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"[error] {exc!r}", flush=True)
                    continue
                stats["returned"] += len(articles)
                for art in articles:
                    url = (art.get("article_url") or "").strip()
                    published = art.get("published_utc")
                    if not url or not published:
                        continue
                    ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    if not in_window(ts):
                        continue
                    stats["in_window"] += 1
                    tickers_of = [t.upper() for t in (art.get("tickers") or [ticker])]
                    if url in rows:
                        rows[url]["tickers"] = sorted(set(rows[url]["tickers"]) | set(tickers_of))
                        continue
                    rows[url] = {
                        "url": url,
                        "url_canonical": canonicalize_url(url),
                        "source_domain": domain_of(url),
                        "publisher": (art.get("publisher") or {}).get("name") or None,
                        "tickers": sorted(set(tickers_of)),
                        "published_utc": ts.astimezone(timezone.utc),
                        "title": art.get("title") or None,
                        "author": art.get("author") or None,
                    }

            if rows and not args.dry_run:
                with conn.cursor() as cur:
                    cur.executemany(INSERT, list(rows.values()))
            stats["inserted"] += len(rows)
            seen.update(rows)

            if done % 10 == 0 or done == len(tickers):
                eta = (time.monotonic() - started) / done * (len(tickers) - done) / 60
                print(f"[progress] {done}/{len(tickers)} tickers :: {stats} "
                      f":: {len(seen)} unique url(s) :: eta {eta:.0f} min", flush=True)

        print(f"[done] {stats} :: {len(seen)} unique url(s)"
              + (" (dry run, nothing written)" if args.dry_run else ""), flush=True)


if __name__ == "__main__":
    main()
