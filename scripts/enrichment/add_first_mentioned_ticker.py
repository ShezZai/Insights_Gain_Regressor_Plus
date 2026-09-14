#!/usr/bin/env python3
"""Add articles.first_mentioned_ticker -- the universe ticker named earliest
in the article itself.

`primary_ticker` is positional: the tag stage takes the first entry of the
`tickers` array as the feed happened to order it, which is why a
Sherwin-Williams story can carry primary_ticker = NVDA. This column is the
text-derived alternative: the article's title and body are scanned with the
same matcher the tag stage uses for `more_tickers` (company names, $CASH,
(NASDAQ:XYZ) forms, and unambiguous bare symbols), and the ticker with the
earliest first mention wins. Title is searched before body, so a headline
mention always outranks one buried in paragraph nine.

Share classes are collapsed to one ticker per company: a mention of "Alphabet"
matches both GOOG and GOOGL at the same position, so the class that sorts first
wins -- which is also the class the news feed uses (GOOG, BELFA). Without this
the same sentence could yield either class, inflating disagreement with
primary_ticker for no real reason.

NULL means no universe ticker was found in the text at all -- a real signal
that the article may not be about any name in the universe.

Only the 118-ticker universe in public.ticker_data is matched; run
`ticker-news load-universe` first if that table is empty.

Usage:
    python add_first_mentioned_ticker.py --dry-run
    python add_first_mentioned_ticker.py
    python add_first_mentioned_ticker.py --reprocess
"""

from __future__ import annotations

import argparse
from collections import Counter

import psycopg

from ticker_news.enrichment.tagging import build_matcher, load_ticker_data
from ticker_news.shared.db import resolve_dsn

DDL = "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS first_mentioned_ticker text"


def build_class_map(data: dict) -> dict:
    """ticker -> the one class representing its company (GOOGL -> GOOG)."""
    by_company: dict = {}
    for ticker, (name, _seg) in data.items():
        key = (name or ticker).strip().lower()
        by_company.setdefault(key, []).append(ticker)
    canonical: dict = {}
    for classes in by_company.values():
        winner = sorted(classes)[0]
        for ticker in classes:
            canonical[ticker] = winner
    return canonical


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default: DATABASE_URL / NEWS_DB_DSN from .env)")
    ap.add_argument("--reprocess", action="store_true",
                    help="Recompute rows that already have a value")
    ap.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    args = ap.parse_args()

    with psycopg.connect(resolve_dsn(args.dsn), autocommit=True) as conn:
        conn.execute(DDL)
        data = load_ticker_data(conn)
        find = build_matcher(data)
        canonical = build_class_map(data)
        collapsed = {k: v for k, v in canonical.items() if k != v}
        print(f"[classes] collapsing {collapsed or 'nothing'}")

        where = "content IS NOT NULL AND char_length(content) > 0"
        if not args.reprocess:
            where += " AND first_mentioned_ticker IS NULL"
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, title, content FROM public.articles WHERE {where} ORDER BY id")
            rows = cur.fetchall()
        print(f"[plan] {len(rows)} article(s) to scan against {len(data)} universe ticker(s)")

        updates: list[tuple] = []
        found = Counter()
        for article_id, title, content in rows:
            hits = find(f"{title or ''}\n{content}")
            first = canonical.get(hits[0], hits[0]) if hits else None
            found[first is not None] += 1
            updates.append((first, article_id))

        print(f"[scan] {found[True]} with a ticker in text, {found[False]} without")
        if args.dry_run:
            print("[dry-run] nothing written")
            return

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE public.articles SET first_mentioned_ticker = %s WHERE id = %s",
                updates,
            )
        print(f"[done] wrote first_mentioned_ticker for {len(updates)} row(s)")


if __name__ == "__main__":
    main()
