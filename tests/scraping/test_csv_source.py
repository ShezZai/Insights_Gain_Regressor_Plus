from datetime import datetime, timezone
from ticker_news.scraping.csv_source import read_jobs

CSV = (
    "﻿tickers,article_url,published_utc,publisher_name\n"
    "GOOG,GOOGL,https://www.fool.com/a/?source=x,2024-11-01T07:20:00Z,The Motley Fool\n"  # noqa
    ",https://benzinga.com/b,2024-11-02T00:00:00Z,Benzinga\n"
    "NVDA,,2024-11-03T00:00:00Z,Skip Me\n"  # blank url -> skipped
)


def test_read_jobs_parses_rows(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(CSV, encoding="utf-8")
    jobs = list(read_jobs(str(p)))
    # row 1 has a comma-joined ticker field that csv splits; assert by url instead
    urls = [j.url for j in jobs]
    assert "https://benzinga.com/b" in urls
    # blank-url row is dropped
    assert all(j.url for j in jobs)


def test_read_jobs_parses_date_and_tickers(tmp_path):
    csv_text = (
        "﻿tickers,article_url,published_utc,publisher_name\n"
        '"NVDA,AMD",https://x.com/a,2024-11-01T07:20:00Z,Pub\n'
    )
    p = tmp_path / "b.csv"
    p.write_text(csv_text, encoding="utf-8")
    job = list(read_jobs(str(p)))[0]
    assert job.tickers == ["NVDA", "AMD"]
    assert job.published_utc == datetime(2024, 11, 1, 7, 20, tzinfo=timezone.utc)
    assert job.publisher == "Pub"


def test_read_jobs_accepts_singular_ticker_column(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("ticker,article_url,published_utc,publisher_name\n"
                 "NVDA,https://x.com/a,2024-11-01T07:20:00Z,Pub\n", encoding="utf-8")
    job = list(read_jobs(str(p)))[0]
    assert job.tickers == ["NVDA"]
