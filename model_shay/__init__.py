"""Text + market-floats regression model (see model_shay_spec.md).

Predicts `gain_90m_after_article` from an article's consolidated insight text
plus three market floats, over the news_trading_window corpus.
"""

__all__ = ["config", "data", "model", "train", "baselines", "predict", "cli"]
