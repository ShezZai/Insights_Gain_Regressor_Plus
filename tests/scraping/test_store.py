from datetime import datetime, timezone
import pytest

pytestmark = pytest.mark.db


def _save(store, url, status="ok", content="hello world body text"):
    store.save(
        url=url, url_canonical=url, source_domain="fool.com", publisher="Fool",
        tickers=["NVDA", "AMD"], published_utc=datetime(2024, 11, 1, tzinfo=timezone.utc),
        title="T", author="A", content=content, raw_html="<html>x</html>", lang="en",
        word_count=3, fetch_method="http", http_status=200, status=status, error=None,
    )


def test_save_then_exists_ok(store):
    assert store.exists_ok("https://fool.com/a") is False
    _save(store, "https://fool.com/a")
    assert store.exists_ok("https://fool.com/a") is True


def test_save_is_idempotent_on_url(store):
    _save(store, "https://fool.com/a", content="first")
    _save(store, "https://fool.com/a", content="second")
    row = store.conn.execute(
        "SELECT count(*), max(content) FROM articles WHERE url=%s", ("https://fool.com/a",)
    ).fetchone()
    assert row[0] == 1          # one row, not two
    assert row[1] == "second"   # upsert overwrote


def test_error_row_is_not_ok(store):
    _save(store, "https://fool.com/b", status="error", content=None)
    assert store.exists_ok("https://fool.com/b") is False
