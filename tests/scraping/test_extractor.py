from ticker_news.scraping.models import RawPage
from ticker_news.scraping.extract.extractor import extract, register, SITE_OVERRIDES

ARTICLE_HTML = """
<html><head><title>NVDA soars on AI demand</title></head><body>
<nav>Home Markets Login Subscribe</nav>
<aside class="ad">BUY NOW! Sponsored content here</aside>
<article>
<h1>NVDA soars on AI demand</h1>
<p>Nvidia reported record data-center revenue this quarter, beating analyst
expectations by a wide margin as demand for AI accelerators continued to surge.</p>
<p>The company guided to further growth next quarter, citing strong orders from
hyperscale cloud customers and new networking products.</p>
<p>Management highlighted supply improvements in advanced packaging that should
ease the bottlenecks that constrained shipments earlier in the year.</p>
</article>
<div class="related">Related: AMD, INTC, AVGO</div>
<footer>Copyright 2024 Example Publisher</footer>
</body></html>
"""


def _raw(html, url="https://www.fool.com/investing/2024/11/01/nvda/"):
    return RawPage(url=url, final_url=url, status=200, html=html, method="http")


def test_generic_extracts_body_and_drops_boilerplate():
    art = extract(_raw(ARTICLE_HTML), min_words=20)
    assert art is not None
    assert "record data-center revenue" in art.text
    assert "advanced packaging" in art.text
    # boilerplate is gone
    assert "Sponsored content" not in art.text
    assert "Login Subscribe" not in art.text
    assert art.word_count >= 40


def test_override_used_when_generic_is_weak():
    # A page generic can't parse (no real article structure).
    junk = _raw("<html><body><div>x</div></body></html>", url="https://example-override.com/a")

    @register("example-override.com")
    def _ov(html: str) -> str:
        return "Overridden body text that is sufficiently long for the threshold." * 2

    try:
        art = extract(junk, min_words=5)
        assert art is not None
        assert "Overridden body text" in art.text
    finally:
        SITE_OVERRIDES.pop("example-override.com", None)
