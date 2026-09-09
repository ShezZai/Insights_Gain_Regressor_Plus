from datetime import datetime
from pathlib import Path

import psycopg

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_INSERT = """
INSERT INTO articles (
    url, url_canonical, source_domain, publisher, tickers, published_utc,
    title, author, content, raw_html, lang, word_count, fetch_method,
    http_status, status, error, fetched_at, extracted_at
) VALUES (
    %(url)s, %(url_canonical)s, %(source_domain)s, %(publisher)s, %(tickers)s, %(published_utc)s,
    %(title)s, %(author)s, %(content)s, %(raw_html)s, %(lang)s, %(word_count)s, %(fetch_method)s,
    %(http_status)s, %(status)s, %(error)s, now(), now()
)
ON CONFLICT (url) DO UPDATE SET
    url_canonical = EXCLUDED.url_canonical,
    source_domain = EXCLUDED.source_domain,
    publisher     = EXCLUDED.publisher,
    tickers       = EXCLUDED.tickers,
    published_utc = EXCLUDED.published_utc,
    title         = EXCLUDED.title,
    author        = EXCLUDED.author,
    content       = EXCLUDED.content,
    raw_html      = EXCLUDED.raw_html,
    lang          = EXCLUDED.lang,
    word_count    = EXCLUDED.word_count,
    fetch_method  = EXCLUDED.fetch_method,
    http_status   = EXCLUDED.http_status,
    status        = EXCLUDED.status,
    error         = EXCLUDED.error,
    fetched_at    = now(),
    extracted_at  = now()
"""


class Store:
    def __init__(self, dsn: str):
        # autocommit: each save is its own transaction -> safe to kill/resume.
        self.conn = psycopg.connect(dsn, autocommit=True)

    def init_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        # psycopg3 uses the extended protocol (one statement per execute), so
        # split the script and run each statement separately.
        for statement in (s.strip() for s in sql.split(";")):
            if statement:
                self.conn.execute(statement)

    def exists_ok(self, url: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE url = %s AND status = 'ok'", (url,)
        ).fetchone()
        return row is not None

    def save(
        self,
        *,
        url: str,
        url_canonical: str | None,
        source_domain: str,
        publisher: str | None,
        tickers: list[str],
        published_utc: datetime | None,
        title: str | None,
        author: str | None,
        content: str | None,
        raw_html: str | None,
        lang: str | None,
        word_count: int,
        fetch_method: str | None,
        http_status: int | None,
        status: str,
        error: str | None,
    ) -> None:
        self.conn.execute(_INSERT, {
            "url": url, "url_canonical": url_canonical, "source_domain": source_domain,
            "publisher": publisher, "tickers": tickers, "published_utc": published_utc,
            "title": title, "author": author, "content": content, "raw_html": raw_html,
            "lang": lang, "word_count": word_count, "fetch_method": fetch_method,
            "http_status": http_status, "status": status, "error": error,
        })

    def close(self) -> None:
        self.conn.close()
