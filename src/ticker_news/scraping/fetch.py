import asyncio

import httpx

from .config import Settings
from .models import RawPage

_CHALLENGE_MARKERS = (
    "Just a moment",
    "cf-browser-verification",
    "Checking your browser",
    "Attention Required",
    "Enable JavaScript and cookies to continue",
)


def http_looks_bad(raw: RawPage | None) -> bool:
    if raw is None:
        return True
    if raw.status in (401, 403, 404, 429) or raw.status >= 500:
        return True
    if not raw.html or len(raw.html) < 800:
        return True
    head = raw.html[:2000]
    return any(marker in head for marker in _CHALLENGE_MARKERS)


class Fetcher:
    """HTTP-first fetcher with a lazily-launched, reused Playwright browser."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._client = httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        )
        self._pw = None
        self._browser = None
        self._ctx = None
        self._browser_lock = asyncio.Lock()

    async def http_get(self, url: str) -> RawPage | None:
        try:
            resp = await self._client.get(url)
        except httpx.HTTPError:
            return None
        return RawPage(url=url, final_url=str(resp.url), status=resp.status_code,
                       html=resp.text, method="http")

    async def _ensure_browser(self) -> None:
        # Lock so concurrent workers don't double-launch (which would leak a
        # playwright subprocess by overwriting self._pw/_browser/_ctx).
        async with self._browser_lock:
            if self._browser is None:
                from playwright.async_api import async_playwright
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=True)
                self._ctx = await self._browser.new_context(user_agent=self.s.user_agent)

    async def browser_get(self, url: str) -> RawPage | None:
        await self._ensure_browser()
        page = await self._ctx.new_page()
        try:
            resp = await page.goto(
                url, wait_until="domcontentloaded",
                timeout=int(self.s.http_timeout_s * 3 * 1000),
            )
            html = await page.content()
            return RawPage(url=url, final_url=page.url,
                           status=resp.status if resp else 0,
                           html=html, method="playwright")
        except Exception:
            return None
        finally:
            await page.close()

    async def aclose(self) -> None:
        await self._client.aclose()
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            if self._pw is not None:
                await self._pw.stop()
