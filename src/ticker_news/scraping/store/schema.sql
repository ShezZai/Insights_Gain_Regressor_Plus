-- scraper/store/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,
    url_canonical TEXT,
    source_domain TEXT NOT NULL,
    publisher     TEXT,
    tickers       TEXT[] NOT NULL DEFAULT '{}',
    published_utc TIMESTAMPTZ,
    title         TEXT,
    author        TEXT,
    content       TEXT,
    raw_html      TEXT,
    lang          TEXT,
    word_count    INT,
    fetch_method  TEXT,
    http_status   INT,
    status        TEXT NOT NULL,
    error         TEXT,
    fetched_at    TIMESTAMPTZ,
    extracted_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS articles_tickers_idx ON articles USING GIN (tickers);
CREATE INDEX IF NOT EXISTS articles_domain_idx ON articles (source_domain);
CREATE INDEX IF NOT EXISTS articles_published_idx ON articles (published_utc);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS provider_sentiments JSONB;
