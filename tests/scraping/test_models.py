from datetime import datetime, timezone
from ticker_news.scraping.models import ArticleJob, RawPage, Article


def test_article_word_count_and_weakness():
    art = Article(title="t", text="one two three four five", author=None,
                  published=None, lang="en")
    assert art.word_count == 5
    assert art.is_weak(min_words=10) is True
    assert art.is_weak(min_words=3) is False


def test_empty_article_is_weak():
    art = Article(title=None, text="", author=None, published=None, lang=None)
    assert art.word_count == 0
    assert art.is_weak(min_words=1) is True


def test_dataclasses_construct():
    job = ArticleJob(url="u", tickers=["NVDA"],
                     published_utc=datetime(2024, 11, 1, tzinfo=timezone.utc), publisher="Fool")
    raw = RawPage(url="u", final_url="u2", status=200, html="<html></html>", method="http")
    assert job.tickers == ["NVDA"] and raw.method == "http"
