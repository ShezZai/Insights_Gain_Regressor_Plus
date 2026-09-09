import pytest

from ticker_news.ingestion.csv_backfill import CsvBackfillSource
from ticker_news.ingestion.feed import FeedItem

CSV = """\
tickers,article_url,published_utc,publisher_name
"NVDA,AMD",https://example.com/a,2026-01-02T03:04:05Z,Benzinga
NVDA,https://example.com/b,,
,,2026-01-01T00:00:00Z,NoUrl
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "news.csv"
    p.write_text(CSV, encoding="utf-8")
    return str(p)


async def _collect(source):
    return [item async for item in source.stream()]


async def test_yields_feed_items_with_parsed_fields(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    assert len(items) == 2  # the url-less row is dropped
    first = items[0]
    assert isinstance(first, FeedItem)
    assert first.url == "https://example.com/a"
    assert first.tickers == ["NVDA", "AMD"]
    assert first.published_utc is not None and first.published_utc.year == 2026
    assert first.publisher == "Benzinga"


async def test_missing_optional_fields_are_none(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    second = items[1]
    assert second.url == "https://example.com/b"
    assert second.published_utc is None
    assert second.publisher is None


async def test_limit_caps_rows(csv_path):
    items = await _collect(CsvBackfillSource(csv_path, limit=1))
    assert len(items) == 1


LEGACY_CSV = """\
ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name
NVDA,https://example.com/a,2026-01-02T03:04:05Z,positive,beats guidance,Benzinga
AMD,https://example.com/a,2026-01-02T03:04:05Z,negative,loses share,Benzinga
NVDA,https://example.com/b,2026-01-03T00:00:00Z,,,Reuters
"""


async def test_legacy_rows_merge_by_url_with_sentiments(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text(LEGACY_CSV, encoding="utf-8")
    items = await _collect(CsvBackfillSource(str(p)))
    assert len(items) == 2
    a = items[0]
    assert a.url == "https://example.com/a"
    assert a.tickers == ["NVDA", "AMD"]
    assert a.source_meta["provider"] == "csv"
    assert a.source_meta["sentiments"] == {
        "NVDA": {"sentiment": "positive", "sentiment_reasoning": "beats guidance"},
        "AMD": {"sentiment": "negative", "sentiment_reasoning": "loses share"},
    }
    b = items[1]
    assert b.source_meta["sentiments"] == {}  # blank sentiment columns -> no entry


async def test_limit_counts_articles_not_rows(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text(LEGACY_CSV, encoding="utf-8")
    items = await _collect(CsvBackfillSource(str(p), limit=1))
    assert len(items) == 1 and items[0].tickers == ["NVDA", "AMD"]
