from ticker_news.scraping.urls import domain_of, canonicalize_url


def test_domain_strips_www_and_lowercases():
    assert domain_of("https://WWW.Fool.com/investing/x") == "fool.com"
    assert domain_of("https://benzinga.com/a/b") == "benzinga.com"


def test_canonicalize_drops_tracking_params():
    url = "https://www.fool.com/investing/2024/11/01/x/?source=iedfolrf0000001&utm_medium=feed"
    assert canonicalize_url(url) == "https://www.fool.com/investing/2024/11/01/x"


def test_canonicalize_keeps_meaningful_query():
    url = "https://www.investing.com/analysis/x-200653661?foo=bar"
    assert canonicalize_url(url) == "https://www.investing.com/analysis/x-200653661?foo=bar"
