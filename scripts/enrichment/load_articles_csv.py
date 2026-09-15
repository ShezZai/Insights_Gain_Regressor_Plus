#!/usr/bin/env python3
"""One-off loader: import a local articles.csv export into a fresh public.articles
table, so enrichment scripts (add_price_features.py, add_extended_gain_features.py)
can run against a local Postgres instance without needing the original DB.

Creates the table (DROP + CREATE -- a one-time bootstrap, not a repeatable
migration) with the CSV's exact column order: `id` as the primary key,
`published_utc` as a real TIMESTAMPTZ, the known numeric gain/feature columns
as DOUBLE PRECISION (so enrichment scripts' NULL-checks and comparisons behave
correctly), and everything else TEXT -- fidelity for a straight CSV round-trip
(load now, enrich, re-export later) doesn't need exact production types.

Loads via Postgres's own COPY (through psycopg3's `cursor.copy()`), which
parses the CSV itself -- including embedded newlines inside quoted text
fields (content, raw_html) that a naive line-based tool would choke on -- so
no pandas round-trip is needed.

Usage:
    python load_articles_csv.py                      # articles.csv -> public.articles
    python load_articles_csv.py --csv other.csv --dsn "dbname=news_trading_window"
"""
from __future__ import annotations

import argparse

import psycopg

# (column, type), in the exact order of the CSV header.
COLUMNS: list[tuple[str, str]] = [
    ("id", "INTEGER PRIMARY KEY"),
    ("url", "TEXT"),
    ("url_canonical", "TEXT"),
    ("source_domain", "TEXT"),
    ("publisher", "TEXT"),
    ("tickers", "TEXT"),
    ("published_utc", "TIMESTAMPTZ"),
    ("title", "TEXT"),
    ("author", "TEXT"),
    ("content", "TEXT"),
    ("raw_html", "TEXT"),
    ("lang", "TEXT"),
    ("word_count", "TEXT"),
    ("fetch_method", "TEXT"),
    ("http_status", "TEXT"),
    ("status", "TEXT"),
    ("error", "TEXT"),
    ("fetched_at", "TEXT"),
    ("extracted_at", "TEXT"),
    ("created_at", "TEXT"),
    ("primary_ticker", "TEXT"),
    ("primary_segment", "TEXT"),
    ("more_tickers", "TEXT"),
    ("more_segments", "TEXT"),
    ("embedding", "TEXT"),
    ("provider_sentiments", "TEXT"),
    ("vix_at_article_time", "DOUBLE PRECISION"),
    ("intraday_gain_till_article", "DOUBLE PRECISION"),
    ("gain_90m_after_article", "DOUBLE PRECISION"),
    ("aiq_ai_etf_intraday_gain_till_article", "DOUBLE PRECISION"),
    ("insights_extracted_at", "TEXT"),
    ("consolidated_insights", "TEXT"),
    ("model_set", "TEXT"),
    ("intraday_gain_till_article_first", "DOUBLE PRECISION"),
    ("gain_90m_after_article_first", "DOUBLE PRECISION"),
    ("first_mentioned_ticker", "TEXT"),
    ("first_listed_ticker", "TEXT"),
    ("more_than_350_insights", "TEXT"),
    ("insights_too_long_more_than_500", "TEXT"),
    ("test_unseen", "TEXT"),
    ("intraday_gain_till_30m_before_article_first", "DOUBLE PRECISION"),
    ("aiq_ai_etf_intraday_gain_till_30m_before_article", "DOUBLE PRECISION"),
    ("vix_intraday_gain_till_30m_before_article_first", "DOUBLE PRECISION"),
    ("article_in_first_market_30m", "TEXT"),
    ("intraday_gain_30m_before_to_90m_after_article_first", "DOUBLE PRECISION"),
    ("intraday_gain_30m_before_till_article_first", "DOUBLE PRECISION"),
    ("category", "TEXT"),
    ("category_reason", "TEXT"),
    ("is_act", "TEXT"),
    ("category_from_insights", "TEXT"),
    ("category_reason_from_insights", "TEXT"),
    ("is_act_from_insights", "TEXT"),
    ("sentiment", "TEXT"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="articles.csv", help="Path to the CSV export")
    ap.add_argument("--dsn", default="dbname=news_trading_window", help="Postgres DSN")
    args = ap.parse_args()

    col_names = [c for c, _ in COLUMNS]
    ddl_cols = ",\n    ".join(f"{c} {t}" for c, t in COLUMNS)

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS public.articles")
        conn.execute(f"CREATE TABLE public.articles (\n    {ddl_cols}\n)")

        col_list = ", ".join(col_names)
        copy_sql = (f"COPY public.articles ({col_list}) FROM STDIN "
                    f"WITH (FORMAT csv, HEADER, NULL '')")
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy, open(args.csv, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    copy.write(chunk)
            print(f"[loaded] {cur.rowcount:,} row(s) into public.articles", flush=True)

        conn.execute("CREATE INDEX IF NOT EXISTS articles_primary_ticker_idx "
                    "ON public.articles (primary_ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS articles_published_utc_idx "
                    "ON public.articles (published_utc)")


if __name__ == "__main__":
    main()
