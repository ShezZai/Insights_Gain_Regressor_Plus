import asyncio
import contextlib
import time

from .config import Settings
from .csv_source import read_jobs
from .extract.extractor import extract
from .fetch import Fetcher, http_looks_bad
from .models import ArticleJob, RawPage
from .robots import RobotsCache
from .store.db import Store
from .urls import canonicalize_url, domain_of


class DomainLimiter:
    """Caps concurrency per domain and enforces a min delay between requests."""

    def __init__(self, per_domain: int, delay: float):
        self.per_domain = per_domain
        self.delay = delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._last: dict[str, float] = {}

    def _sem(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._sems:
            self._sems[domain] = asyncio.Semaphore(self.per_domain)
        return self._sems[domain]

    @contextlib.asynccontextmanager
    async def slot(self, domain: str):
        sem = self._sem(domain)
        await sem.acquire()
        try:
            if self.delay > 0:
                wait = self.delay - (time.monotonic() - self._last.get(domain, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()
            yield
        finally:
            sem.release()


async def _http_with_retries(fetcher, url: str, settings: Settings, attempts: int = 3) -> RawPage | None:
    raw = None
    delay = 1.0
    for attempt in range(attempts):
        raw = await fetcher.http_get(url)
        if raw is not None and raw.status != 429 and raw.status < 500:
            return raw
        if attempt < attempts - 1:  # don't sleep after the final attempt
            await asyncio.sleep(delay)
            delay *= 2
    return raw


async def process_job(job: ArticleJob, fetcher, store, settings: Settings,
                      limiter: DomainLimiter, robots) -> str:
    domain = domain_of(job.url)

    if settings.respect_robots and robots is not None:
        if not await asyncio.to_thread(robots.allowed, job.url):
            await asyncio.to_thread(
                store.save, url=job.url, url_canonical=canonicalize_url(job.url),
                source_domain=domain, publisher=job.publisher, tickers=job.tickers,
                published_utc=job.published_utc, title=None, author=None, content=None,
                raw_html=None, lang=None, word_count=0, fetch_method=None, http_status=None,
                status="error", error="blocked_by_robots",
            )
            return "error"

    async with limiter.slot(domain):
        raw = await _http_with_retries(fetcher, job.url, settings)

    used_browser = False
    art = None if (raw is None or http_looks_bad(raw)) else extract(raw, settings.min_words)

    if raw is None or http_looks_bad(raw) or art is None or art.is_weak(settings.min_words):
        async with limiter.slot(domain):
            raw2 = await fetcher.browser_get(job.url)
        # Only adopt the browser result if it's actually usable; otherwise keep
        # the original HTTP response so we don't store a worse page (e.g. a
        # playwright 403 challenge clobbering a real 200 body).
        if raw2 is not None and not http_looks_bad(raw2):
            used_browser = True
            raw = raw2
            art = extract(raw2, settings.min_words)

    if raw is None:
        status, error, art = "error", "fetch_failed", None
    elif art is None or not art.text.strip():
        status, error = "empty", None
    else:
        status, error = "ok", None

    fetch_method = ("playwright" if used_browser else (raw.method if raw else None))

    await asyncio.to_thread(
        store.save, url=job.url, url_canonical=canonicalize_url(job.url),
        source_domain=domain, publisher=job.publisher, tickers=job.tickers,
        published_utc=job.published_utc,
        title=(art.title if art else None), author=(art.author if art else None),
        content=(art.text if art else None), raw_html=(raw.html if raw else None),
        lang=(art.lang if art else None), word_count=(art.word_count if art else 0),
        fetch_method=fetch_method, http_status=(raw.status if raw else None),
        status=status, error=error,
    )
    return status


async def run(csv_path: str, settings: Settings, limit: int | None = None,
              retry_errors: bool = False) -> dict[str, int]:
    store = Store(settings.db_dsn)
    store.init_schema()
    fetcher = Fetcher(settings)
    robots = RobotsCache(settings.user_agent) if settings.respect_robots else None
    limiter = DomainLimiter(settings.per_domain, settings.domain_delay_s)

    counts = {"ok": 0, "empty": 0, "error": 0, "skip": 0}
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.concurrency * 4)

    async def worker():
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                if not retry_errors and await asyncio.to_thread(store.exists_ok, job.url):
                    counts["skip"] += 1
                else:
                    res = await process_job(job, fetcher, store, settings, limiter, robots)
                    counts[res] = counts.get(res, 0) + 1
                done = sum(counts.values())
                if done % 50 == 0:
                    print(f"[progress] {done} done :: {counts}")
            except Exception as exc:  # never let one bad URL kill a worker
                counts["error"] = counts.get("error", 0) + 1
                print(f"[worker error] {getattr(job, 'url', '?')}: {exc!r}")
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(settings.concurrency)]

    try:
        produced = 0
        for job in read_jobs(csv_path):
            if limit is not None and produced >= limit:
                break
            produced += 1
            await queue.put(job)
        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)
    finally:
        # Always release the browser subprocess, httpx client, and DB connection,
        # even on Ctrl+C or an unexpected error mid-run.
        for w in workers:
            w.cancel()
        await fetcher.aclose()
        store.close()
    print(f"[done] {counts}")
    return counts
