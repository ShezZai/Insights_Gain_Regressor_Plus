"""Gradient-boosted tree over the dense features only (no text encoder).

The counterpart to model_shay: same corpus, same splits, same GroupKFold, same
metric definitions -- but the 10-wide dense vector is the whole input. What the
text contributes is exactly the gap between the two.
"""

__all__ = ["tree", "cli"]
