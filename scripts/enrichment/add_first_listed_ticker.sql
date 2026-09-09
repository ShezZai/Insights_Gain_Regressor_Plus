-- articles.first_listed_ticker: the first ticker in the feed's `tickers` array
-- that belongs to the universe (public.ticker_data), preserving array order.
--
-- This is the feed-derived counterpart to first_mentioned_ticker (text-derived).
-- It exists because 534 scraped articles name no universe ticker anywhere in
-- their text, yet every one of them carries a universe ticker in the array --
-- so it is the only usable fallback for those rows:
--
--     COALESCE(first_mentioned_ticker, first_listed_ticker)
--
-- Note it is NOT the same as primary_ticker, which is plain `tickers[0]` and so
-- can be a non-universe symbol (AAPL, BRK.A ...) whenever the publisher's tag
-- list is alphabetical rather than relevance-ordered.
--
-- Idempotent: re-running recomputes every row.
--
--   psql -d news_trading_window -f add_first_listed_ticker.sql

ALTER TABLE public.articles
    ADD COLUMN IF NOT EXISTS first_listed_ticker text;

UPDATE public.articles a
SET first_listed_ticker = pick.ticker
FROM (
    SELECT src.id, first_universe.ticker
    FROM public.articles src
    CROSS JOIN LATERAL (
        SELECT u.ticker
        FROM unnest(src.tickers) WITH ORDINALITY AS u(ticker, ord)
        WHERE u.ticker IN (SELECT ticker FROM public.ticker_data)
        ORDER BY u.ord
        LIMIT 1
    ) AS first_universe
) AS pick
WHERE a.id = pick.id
  AND a.first_listed_ticker IS DISTINCT FROM pick.ticker;
