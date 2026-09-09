import threading
import urllib.robotparser
from urllib.parse import urlsplit

import httpx


class RobotsCache:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        # allowed() is called from worker threads (asyncio.to_thread); the lock
        # makes the check-then-fetch atomic so robots.txt is fetched once per host.
        self._lock = threading.Lock()

    def _fetch_robots_text(self, robots_url: str) -> str | None:
        try:
            resp = httpx.get(
                robots_url, timeout=10, follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        except httpx.HTTPError:
            return None
        return resp.text if resp.status_code == 200 else None

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        key = parts.netloc
        with self._lock:
            parser = self._cache.get(key)
            if parser is None:
                parser = urllib.robotparser.RobotFileParser()
                text = self._fetch_robots_text(f"{parts.scheme}://{parts.netloc}/robots.txt")
                parser.parse((text or "").splitlines())  # empty -> allow all
                self._cache[key] = parser
        return parser.can_fetch(self.user_agent, url)
