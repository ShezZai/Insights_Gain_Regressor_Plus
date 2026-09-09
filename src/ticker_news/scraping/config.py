from dataclasses import dataclass, field

from ticker_news.shared.config import get_settings


@dataclass(frozen=True)
class Settings:
    """Frozen per-run scraper settings.

    Defaults come from the unified AppSettings (DATABASE_URL, SCRAPER_* env
    vars); tests and the CLI override individual fields via dataclasses.replace.
    """

    db_dsn: str = field(default_factory=lambda: get_settings().database_url)
    concurrency: int = field(default_factory=lambda: get_settings().scraper_concurrency)
    per_domain: int = field(default_factory=lambda: get_settings().scraper_per_domain)
    domain_delay_s: float = field(default_factory=lambda: get_settings().scraper_domain_delay_s)
    http_timeout_s: float = field(default_factory=lambda: get_settings().scraper_http_timeout_s)
    min_words: int = field(default_factory=lambda: get_settings().scraper_min_words)
    respect_robots: bool = field(default_factory=lambda: get_settings().scraper_respect_robots)
    user_agent: str = field(default_factory=lambda: get_settings().scraper_user_agent)
