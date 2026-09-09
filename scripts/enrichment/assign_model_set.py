#!/usr/bin/env python3
"""Assign articles.model_set = training | validation | test (60/20/20).

Two constraints pull against each other, so the split is built in that order:

1. Test dates should be unseen during training. The most recent trading days
   are reserved wholesale as a temporal holdout -- every article on those days
   is test, so a model cannot learn the day and replay it.
2. Tickers stay balanced. The holdout alone would skew toward whatever names
   were in the news that month, so each ticker's test quota is topped up from
   the earlier pool until it holds ~20% of that ticker's articles, and its
   remaining articles split 20/60 validation/training.

`--holdout-share` is the fraction of the corpus taken as the date holdout
(default 0.16 = 80% of a 20% test set), leaving the rest of test to the
per-ticker top-up. Sampling is seeded, so re-running reproduces the split.

Balancing uses COALESCE(first_mentioned_ticker, first_listed_ticker) -- always
a universe ticker, and set for every row -- rather than primary_ticker, which
is `tickers[0]` from the feed and can be a non-universe symbol (AAPL, BRK.A).
Pass --ticker-column primary for the old behaviour.

Only rows with consolidated_insights are split: a row with no insight text
cannot feed the model, and counting it would make the 60/20/20 describe rows
that never reach training. --all-rows includes everything instead; rows left
out get model_set = NULL.

Usage:
    python assign_model_set.py                 # write the split
    python assign_model_set.py --dry-run       # report only, no writes
    python assign_model_set.py --seed 7
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

import psycopg

ET = ZoneInfo("America/New_York")
TRAIN, VAL, TEST = "training", "validation", "test"

DDL = "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS model_set text"


def mark_test_unseen(conn) -> None:
    """articles.test_unseen: true for test rows whose CALENDAR DAY appears in
    neither training nor validation -- the part of test a model cannot have
    seen the market conditions of. NULL outside the test split."""
    conn.execute("ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS test_unseen boolean")
    conn.execute("UPDATE public.articles SET test_unseen = NULL WHERE test_unseen IS NOT NULL")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE public.articles a
            SET test_unseen = NOT EXISTS (
                SELECT 1 FROM public.articles b
                WHERE b.model_set IN ('training', 'validation')
                  AND (b.published_utc AT TIME ZONE 'America/New_York')::date
                    = (a.published_utc AT TIME ZONE 'America/New_York')::date
            )
            WHERE a.model_set = 'test'
        """)
        print(f"[test_unseen] marked {cur.rowcount} test row(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="dbname=news_trading_window")
    ap.add_argument("--test-share", type=float, default=0.20)
    ap.add_argument("--val-share", type=float, default=0.20)
    ap.add_argument("--holdout-share", type=float, default=0.16,
                    help="Fraction of the corpus reserved as the trailing date holdout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticker-column", default="COALESCE(first_mentioned_ticker, first_listed_ticker)",
                    help="expression to balance on; 'primary' selects primary_ticker")
    ap.add_argument("--all-rows", action="store_true",
                    help="split every row, not just those with consolidated_insights")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(DDL)
        ticker_expr = ("primary_ticker" if args.ticker_column == "primary"
                       else args.ticker_column)
        clauses = [f"({ticker_expr}) IS NOT NULL", "published_utc IS NOT NULL"]
        if not args.all_rows:
            clauses.append("consolidated_insights IS NOT NULL")
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, {ticker_expr}, "
                f"(published_utc AT TIME ZONE 'America/New_York')::date "
                f"FROM public.articles WHERE {' AND '.join(clauses)} ORDER BY id"
            )
            rows = cur.fetchall()
        # Rows outside the split (no insights) must not keep a stale label.
        if not args.dry_run:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE public.articles SET model_set = NULL "
                            f"WHERE NOT ({' AND '.join(clauses)})")
                print(f"[reset] cleared model_set on {cur.rowcount} unmodelable row(s)")

        total = len(rows)
        # 1. Trailing date holdout: whole days, newest first, until the quota.
        per_day = Counter(day for _, _, day in rows)
        holdout_days: set = set()
        taken = 0
        for day in sorted(per_day, reverse=True):
            if taken >= args.holdout_share * total:
                break
            holdout_days.add(day)
            taken += per_day[day]

        assignment: dict[int, str] = {}
        early_by_ticker: dict[str, list[int]] = defaultdict(list)
        counts_by_ticker: Counter = Counter()
        test_by_ticker: Counter = Counter()

        for article_id, ticker, day in rows:
            counts_by_ticker[ticker] += 1
            if day in holdout_days:
                assignment[article_id] = TEST
                test_by_ticker[ticker] += 1
            else:
                early_by_ticker[ticker].append(article_id)

        # 2. Per-ticker top-up so every name lands near the target shares.
        for ticker, pool in early_by_ticker.items():
            rng.shuffle(pool)
            n_ticker = counts_by_ticker[ticker]
            want_test = max(0, round(args.test_share * n_ticker) - test_by_ticker[ticker])
            want_val = round(args.val_share * n_ticker)
            want_test = min(want_test, len(pool))
            want_val = min(want_val, len(pool) - want_test)
            for i, article_id in enumerate(pool):
                if i < want_test:
                    assignment[article_id] = TEST
                elif i < want_test + want_val:
                    assignment[article_id] = VAL
                else:
                    assignment[article_id] = TRAIN

        split = Counter(assignment.values())
        print(f"[split] {total} article(s) over {len(per_day)} trading day(s)")
        for name in (TRAIN, VAL, TEST):
            print(f"  {name:11} {split[name]:5}  {split[name] / total:6.1%}")
        print(f"[holdout] {len(holdout_days)} day(s) from {min(holdout_days)} onward "
              f"= {taken} article(s), {taken / max(split[TEST], 1):.1%} of test")

        if args.dry_run:
            print("[dry-run] nothing written")
            return

        with conn.cursor() as cur:
            cur.executemany("UPDATE public.articles SET model_set = %s WHERE id = %s",
                            [(v, k) for k, v in assignment.items()])
        print(f"[done] wrote model_set for {len(assignment)} row(s)")
        mark_test_unseen(conn)


if __name__ == "__main__":
    main()
