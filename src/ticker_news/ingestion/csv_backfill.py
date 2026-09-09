"""CSV backfill source — replays a news CSV (legacy per-ticker rows or merged
rows) as a NewsFeedSource, carrying provider sentiment into source_meta."""

from __future__ import annotations

import csv
from datetime import datetime
from typing import AsyncIterator

from ticker_news.ingestion.feed import FeedItem


def _parse_dt(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_feed_items(path: str) -> list[FeedItem]:
    """Group CSV rows by URL (legacy CSVs carry one row per ticker-url pair).

    Tickers merge in first-seen order; non-blank sentiment columns become the
    same source_meta["sentiments"] shape the Massive poller produces.
    """
    merged: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("article_url") or "").strip()
            if not url:
                continue
            entry = merged.setdefault(url, {
                "tickers": [], "sentiments": {},
                "published_utc": _parse_dt(row.get("published_utc")),
                "publisher": (row.get("publisher_name") or "").strip() or None,
            })
            raw = row.get("tickers") or row.get("ticker") or ""
            for t in (x.strip().upper() for x in raw.split(",")):
                if t and t not in entry["tickers"]:
                    entry["tickers"].append(t)
                    sentiment = (row.get("sentiment") or "").strip()
                    if sentiment:
                        entry["sentiments"][t] = {
                            "sentiment": sentiment,
                            "sentiment_reasoning": (row.get("sentiment_reasoning") or "").strip(),
                        }
    return [
        FeedItem(
            url=url,
            tickers=e["tickers"],
            published_utc=e["published_utc"],
            publisher=e["publisher"],
            source_meta={"sentiments": e["sentiments"], "provider": "csv"},
        )
        for url, e in merged.items()
    ]


class CsvBackfillSource:
    def __init__(self, csv_path: str, *, limit: int | None = None):
        self.csv_path = csv_path
        self.limit = limit

    async def stream(self) -> AsyncIterator[FeedItem]:
        # Blocking file I/O is intentional: backfill reads one bounded CSV at
        # startup. A high-throughput source should use asyncio.to_thread.
        items = read_feed_items(self.csv_path)
        for i, item in enumerate(items):
            if self.limit is not None and i >= self.limit:
                return
            yield item
