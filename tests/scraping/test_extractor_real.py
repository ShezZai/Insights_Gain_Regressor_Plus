from pathlib import Path
import pytest

from ticker_news.scraping.models import RawPage
from ticker_news.scraping.extract.extractor import extract

FIX = Path(__file__).parent / "fixtures"
PUBLISHERS = ["fool.com", "globenewswire.com", "benzinga.com",
              "investing.com", "marketwatch.com", "zacks.com"]


@pytest.mark.parametrize("domain", PUBLISHERS)
def test_extracts_real_article(domain):
    path = FIX / f"{domain}.html"
    if not path.exists():
        pytest.skip(f"no fixture for {domain}")
    html = path.read_text(encoding="utf-8")
    raw = RawPage(url=f"https://{domain}/a", final_url=f"https://{domain}/a",
                  status=200, html=html, method="http")
    art = extract(raw, min_words=120)
    assert art is not None, f"{domain}: extractor returned None"
    assert art.word_count >= 120, f"{domain}: only {art.word_count} words"
