from ticker_news.scraping.robots import RobotsCache


def test_parses_rules_without_network(monkeypatch):
    cache = RobotsCache(user_agent="test-bot")
    # Stub the network fetch so the test is deterministic/offline.
    monkeypatch.setattr(
        cache, "_fetch_robots_text",
        lambda url: "User-agent: *\nDisallow: /private/\n",
    )
    assert cache.allowed("https://site.com/public/a") is True
    assert cache.allowed("https://site.com/private/secret") is False


def test_missing_robots_allows_all(monkeypatch):
    cache = RobotsCache(user_agent="test-bot")
    monkeypatch.setattr(cache, "_fetch_robots_text", lambda url: None)
    assert cache.allowed("https://nope.com/anything") is True
