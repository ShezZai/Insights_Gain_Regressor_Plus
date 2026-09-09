import os
import urllib.parse

import psycopg
import pytest

TEST_DSN = os.environ.get(
    "TICKER_NEWS_TEST_DSN", "postgresql://scraper:scraper@localhost:5432/news_test"
)


def _db_name(dsn: str) -> str:
    return urllib.parse.urlparse(dsn).path.lstrip("/")


def _admin_dsn(dsn: str) -> str:
    """Same server as TEST_DSN, but the default maintenance database."""
    parts = urllib.parse.urlparse(dsn)
    return urllib.parse.urlunparse(parts._replace(path="/postgres"))


def _connect_test_db():
    """Connect to news_test, creating the database on first run."""
    if "news_test" not in _db_name(TEST_DSN):
        raise RuntimeError(
            f"Refusing to run db tests against {TEST_DSN!r}: the test database "
            "name must contain 'news_test' (this fixture TRUNCATEs tables)."
        )
    try:
        return psycopg.connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        if "news_test" not in str(exc):
            pytest.skip("Postgres not reachable; run `docker compose up -d`")
        admin = psycopg.connect(_admin_dsn(TEST_DSN), autocommit=True)
        admin.execute("CREATE DATABASE news_test")
        admin.close()
        return psycopg.connect(TEST_DSN)


@pytest.fixture
def store():
    from ticker_news.scraping.store.db import Store

    try:
        conn = _connect_test_db()
        conn.close()
        s = Store(TEST_DSN)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    s.init_schema()
    s.conn.execute("TRUNCATE articles CASCADE")
    yield s
    s.conn.execute("TRUNCATE articles CASCADE")
    s.close()
