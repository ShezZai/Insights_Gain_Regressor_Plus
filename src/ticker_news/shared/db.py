import psycopg

from ticker_news.shared.config import get_settings


def connect(*, vector: bool = False, autocommit: bool = False) -> psycopg.Connection:
    """One connection convention for every stage (single DATABASE_URL).

    Default is transactional — callers must commit. Pass autocommit=True for
    one-statement-per-transaction semantics (the scraper-store convention).
    vector=True registers the pgvector adapter (needed to bind numpy arrays
    to vector columns).
    """
    conn = psycopg.connect(get_settings().database_url, autocommit=autocommit)
    if vector:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    return conn
