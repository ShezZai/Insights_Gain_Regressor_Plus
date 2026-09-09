from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Single source of configuration for the whole package.

    One database (DATABASE_URL) for every stage — the scraper's old
    SCRAPER_DB_DSN and the analysis scripts' NEWS_DB_DSN both die here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://scraper:scraper@localhost:5432/news",
        # SCRAPER_DB_DSN is honored as a fallback for the migration window
        # (legacy scraper convention); remove the alias in the final cleanup phase.
        validation_alias=AliasChoices("DATABASE_URL", "SCRAPER_DB_DSN"),
    )

    massive_api_key: str | None = Field(default=None, validation_alias="MASSIVE_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # Scraper knobs — legacy SCRAPER_* env names preserved on purpose.
    scraper_concurrency: int = Field(default=8, validation_alias="SCRAPER_CONCURRENCY")
    scraper_per_domain: int = Field(default=2, validation_alias="SCRAPER_PER_DOMAIN")
    scraper_domain_delay_s: float = Field(default=1.0, validation_alias="SCRAPER_DOMAIN_DELAY")
    scraper_http_timeout_s: float = Field(default=20.0, validation_alias="SCRAPER_HTTP_TIMEOUT")
    scraper_min_words: int = Field(default=120, validation_alias="SCRAPER_MIN_WORDS")
    scraper_respect_robots: bool = Field(default=True, validation_alias="SCRAPER_RESPECT_ROBOTS")
    scraper_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; AITickerNewsBot/0.1; research project)",
        validation_alias="SCRAPER_UA",
    )

    # Historical-precedent retrieval for the sentiment analyst panel.
    # "article":          top-k cosine-nearest prior articles (default, legacy).
    # "insights":         per-insight-box neighbors over public.article_insights.
    # "distilled-first":  same over distilled_article_insights, tagged by first_label.
    # "distilled-second": same, tagged by second_label, with DROP-labelled excluded.
    precedent_source: str = Field(
        default="article", validation_alias="SENTIMENT_PRECEDENT_SOURCE"
    )
    precedent_insights_threshold: float = Field(
        default=0.7, validation_alias="SENTIMENT_PRECEDENT_INSIGHTS_THRESHOLD"
    )
    precedent_insights_limit: int = Field(
        default=60, validation_alias="SENTIMENT_PRECEDENT_INSIGHTS_LIMIT"
    )
    # Lower time bound on the precedent search (all modes): only consider sources
    # published within this many days BEFORE the target. <= 0 means unlimited.
    precedent_lookback_days: int = Field(
        default=90, validation_alias="SENTIMENT_PRECEDENT_LOOKBACK_DAYS"
    )
    # When True, precedents are restricted to category='real news' sources. Default
    # False: an insight is an insight regardless of its source article's category,
    # so recap/analysis/etc. articles are eligible precedents too.
    precedent_real_news_only: bool = Field(
        default=False, validation_alias="SENTIMENT_PRECEDENT_REAL_NEWS_ONLY"
    )

    # Gemini model that generates the buy/sell/hold verdict (the single
    # historical-precedent verdict node). Default is flash-lite for cheap
    # testing; set to "gemini-2.5-flash" for the stronger model. Literal string
    # (not the llm.GEMINI_* constant) to avoid a config<-llm import cycle.
    sentiment_verdict_model: str = Field(
        default="gemini-2.5-flash-lite", validation_alias="SENTIMENT_VERDICT_MODEL"
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
