-- articles.consolidated_insights: every insight for an article as one plain-text
-- block, quotes stripped (article_insights.insight only, never box_text, which
-- carries the QUOTES section). Each block opens with a line naming the article's
-- title and publisher, so the text stands alone as an LLM input.
--
-- Idempotent: re-running rebuilds the column from the current article_insights
-- rows. Articles whose extraction yielded no boxes stay NULL.
--
--   psql -d news_trading_window -f build_consolidated_insights.sql

ALTER TABLE public.articles
    ADD COLUMN IF NOT EXISTS consolidated_insights text;

UPDATE public.articles a
SET consolidated_insights = NULL
WHERE consolidated_insights IS NOT NULL;

UPDATE public.articles a
SET consolidated_insights = b.header || E'\n' || b.body
FROM (
    SELECT i.article_id,
           format(
               'these are the insights related to the article "%s" from %s',
               coalesce(nullif(btrim(art.title), ''), art.url),
               coalesce(nullif(btrim(art.publisher), ''), art.source_domain)
           ) AS header,
           string_agg('- ' || btrim(i.insight), E'\n' ORDER BY i.box_index) AS body
    FROM public.article_insights i
    JOIN public.articles art ON art.id = i.article_id
    WHERE i.insight IS NOT NULL AND btrim(i.insight) <> ''
    GROUP BY i.article_id, art.title, art.url, art.publisher, art.source_domain
) b
WHERE a.id = b.article_id;
