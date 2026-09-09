import csv
from collections.abc import Iterator
from datetime import datetime

from .models import ArticleJob


def _parse_dt(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_jobs(path: str) -> Iterator[ArticleJob]:
    # utf-8-sig strips the BOM present on the CSV header (﻿ticker).
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = (row.get("article_url") or "").strip()
            if not url:
                continue
            tickers = [t.strip() for t in (row.get("tickers") or row.get("ticker") or "").split(",") if t.strip()]
            yield ArticleJob(
                url=url,
                tickers=tickers,
                published_utc=_parse_dt(row.get("published_utc")),
                publisher=(row.get("publisher_name") or "").strip() or None,
            )
