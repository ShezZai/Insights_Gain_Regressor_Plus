#!/usr/bin/env python3
"""Dump one (or more) articles to a folder of plain-text files.

Creates, per article id:

    <out>/article_<id>/
        article.txt              # metadata header + full scraped content
        insights/                # public.article_insights, one file per box
            box_01_<topic>.txt
            ...
        distilled_insights/      # public.distilled_article_insights (if present)
            box_01_<topic>.txt   # includes first_label / second_label
            ...

Handy for eyeballing what the pipeline actually stored for an article.

Usage:
    python dump_article.py 8721
    python dump_article.py 8721 9491 --out /tmp/dumps
    python dump_article.py 8721 --source distilled
    python dump_article.py 8721 --dsn "postgresql://shay@/other?host=/var/run/postgresql"

Defaults to the .env database (DATABASE_URL, else NEWS_DB_DSN) like every other
script here. It used to hardcode `dbname=news` and ignore NEWS_DB_DSN, from when
that name pointed at a different database; it now points at the same corpus, so
the carve-out is gone. Pass --dsn to reach anything else.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import psycopg

from ticker_news.shared.db import resolve_dsn

ARTICLE_COLS = [
    "id", "url", "source_domain", "publisher", "published_utc", "title", "author",
    "tickers", "primary_ticker", "primary_segment", "more_tickers", "more_segments",
    "status", "word_count", "lang", "fetch_method", "fetched_at",
]


def slug(text: str | None, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:maxlen] or "untitled"


def fmt_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def write_article(conn, article_id: int, dest: Path) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(ARTICLE_COLS)} FROM public.articles WHERE id = %s",
            (article_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta = dict(zip(ARTICLE_COLS, row))
        cur.execute("SELECT content FROM public.articles WHERE id = %s", (article_id,))
        content = cur.fetchone()[0]

    dest.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {fmt_value(val)}" for key, val in meta.items()]
    lines += ["", "=" * 78, "", content or "(no content stored)"]
    (dest / "article.txt").write_text("\n".join(lines), encoding="utf-8")
    return meta


def table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
        return cur.fetchone()[0] is not None


def write_insights(conn, article_id: int, dest: Path, table: str, labelled: bool) -> int:
    extra = ", first_label, second_label" if labelled else ""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT box_index, topic, insight, quotes, box_text, model{extra} "
            f"FROM public.{table} WHERE article_id = %s ORDER BY box_index",
            (article_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for row in rows:
        box_index, topic, insight, quotes, box_text, model = row[:6]
        parts = [
            f"article_id: {article_id}",
            f"box_index: {box_index}",
            f"topic: {topic or ''}",
            f"model: {model or ''}",
        ]
        if labelled:
            parts += [f"first_label: {row[6] or ''}", f"second_label: {row[7] or ''}"]
        parts += ["", "--- insight ---", insight or ""]
        if quotes:
            parts += ["", "--- quotes ---"] + [f"- {q}" for q in quotes]
        parts += ["", "--- box_text ---", box_text or ""]
        name = f"box_{box_index:02d}_{slug(topic)}.txt"
        (dest / name).write_text("\n".join(parts), encoding="utf-8")
    return len(rows)


def dump(conn, article_id: int, out_root: Path, source: str) -> None:
    dest = out_root / f"article_{article_id}"
    meta = write_article(conn, article_id, dest)
    if meta is None:
        print(f"[{article_id}] not found", file=sys.stderr)
        return
    print(f"[{article_id}] {dest}/article.txt  ({meta.get('title') or 'no title'})")

    if source in ("article", "both"):
        n = write_insights(conn, article_id, dest / "insights", "article_insights", False)
        print(f"[{article_id}] insights/: {n} box(es)")

    if source in ("distilled", "both"):
        if table_exists(conn, "distilled_article_insights"):
            n = write_insights(
                conn, article_id, dest / "distilled_insights",
                "distilled_article_insights", True,
            )
            print(f"[{article_id}] distilled_insights/: {n} box(es)")
        elif source == "distilled":
            print("distilled_article_insights table not present in this DB", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", type=int, help="article id(s) to dump")
    ap.add_argument("--out", default="article_dumps", help="output root dir")
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default: DATABASE_URL / NEWS_DB_DSN from .env)")
    ap.add_argument("--source", choices=["article", "distilled", "both"], default="both",
                    help="which insight table(s) to dump (default: both)")
    args = ap.parse_args()

    out_root = Path(args.out)
    with psycopg.connect(resolve_dsn(args.dsn)) as conn:
        for article_id in args.ids:
            dump(conn, article_id, out_root, args.source)


if __name__ == "__main__":
    main()
