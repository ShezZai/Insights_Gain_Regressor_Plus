#!/usr/bin/env python3
"""Flag articles whose consolidated_insights exceeds a token budget.

Writes a boolean column (--column): true when the block is longer than
--threshold, false when it fits, NULL when there are no insights. The corpus
carries two of these, because the thresholds are not equivalent -- these blocks
run ~1.28 tokens per word, so 350 words lands nearer 448 tokens than 500:

    more_than_350_insights           > 350 words   (132 rows)
    insights_too_long_more_than_500  > 500 tokens  (81 rows, a strict subset)

Tokens are counted with tiktoken's cl100k_base -- the encoding behind
text-embedding-3-small and the GPT-4 family. Gemini tokenizes differently, so
treat the boundary as approximate if the consumer is Gemini.

Usage:
    python flag_long_insights.py --unit words --threshold 350
    python flag_long_insights.py --unit tokens --threshold 500 \
        --column insights_too_long_more_than_500
"""

from __future__ import annotations

import argparse

import psycopg

from ticker_news.shared.db import resolve_dsn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default: DATABASE_URL / NEWS_DB_DSN from .env)")
    ap.add_argument("--threshold", type=int, default=350)
    ap.add_argument("--unit", choices=["tokens", "words"], default="words")
    ap.add_argument("--column", default="more_than_350_insights",
                    help="boolean column to write (created if missing)")
    args = ap.parse_args()

    if args.unit == "tokens":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        measure = lambda text: len(enc.encode(text))
    else:
        measure = lambda text: len(text.split())

    if not args.column.replace("_", "").isalnum():
        ap.error("--column must be a plain identifier")

    with psycopg.connect(resolve_dsn(args.dsn), autocommit=True) as conn:
        conn.execute(f"ALTER TABLE public.articles "
                     f"ADD COLUMN IF NOT EXISTS {args.column} boolean")
        with conn.cursor() as cur:
            cur.execute("SELECT id, consolidated_insights FROM public.articles "
                        "WHERE consolidated_insights IS NOT NULL")
            rows = cur.fetchall()

        updates = [(measure(text) > args.threshold, article_id) for article_id, text in rows]
        flagged = sum(1 for over, _ in updates if over)
        with conn.cursor() as cur:
            cur.executemany(f"UPDATE public.articles SET {args.column} = %s WHERE id = %s",
                            updates)

        print(f"[done] {args.column}: {len(rows)} row(s) measured in {args.unit} "
              f"(threshold {args.threshold}): {flagged} over, {len(rows) - flagged} under")


if __name__ == "__main__":
    main()
