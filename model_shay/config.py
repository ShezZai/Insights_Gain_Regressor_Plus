"""Run configuration for the text + market-floats regressor.

One dataclass mirrors §9 of model_shay_spec.md. Values arrive in three layers,
each overriding the last:

    dataclass defaults  ->  --config YAML  ->  explicit CLI flags

Every CLI override defaults to None precisely so an unset flag never clobbers a
value the YAML set. Values are coerced to the dataclass field's type on load,
which catches the YAML 1.1 trap where `1e-3` parses as the string "1e-3"
(PyYAML wants `1.0e-3`) -- otherwise a run trains at a garbage LR in silence.

`Config.save()` writes the resolved config next to every checkpoint, so a run
is reproducible from its own output directory (§8).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "default.yaml"

# Which measurement window the features and target come from (§1).
#   at_article  everything measured AT the article's minute; target runs from
#               there to +90m.
#   30m_before  everything measured 30 minutes BEFORE the article; target runs
#               from -30m to +90m, so it also captures whatever the market did
#               in the half hour before publication. These columns exist only
#               in the first_mentioned_ticker flavour.
WINDOW_COLUMNS = {
    "at_article": {
        "float_1": "vix_at_article_time",
        "float_2": "intraday_gain_till_article",
        "float_3": "aiq_ai_etf_intraday_gain_till_article",
        "gain_90m_after": "gain_90m_after_article",
        "intraday_gain_till": "intraday_gain_till_article",
        "sentiment": "sentiment",
    },
    "30m_before": {
        "float_1": "vix_intraday_gain_till_30m_before_article_first",
        "float_2": "intraday_gain_till_30m_before_article_first",
        "float_3": "aiq_ai_etf_intraday_gain_till_30m_before_article",
        "gain_90m_after": "intraday_gain_30m_before_to_90m_after_article_first",
        # The move during the 30 minutes leading up to publication. Deliberately
        # NOT intraday_gain_till_article_first (open -> article): that one
        # correlates 0.95 with float_2 (open -> -30m), so the floats alone would
        # explain 91% of it. This segment correlates 0.006 with float_2, which
        # makes the floats a legitimate input rather than the answer.
        "intraday_gain_till": "intraday_gain_30m_before_till_article_first",
        "sentiment": "sentiment",   # window-independent: a property of the text
    },
}
# Market-wide columns in the at_article window: no per-ticker flavour exists.
# (Every 30m_before name is already final, so the suffix rule skips that window
# entirely -- see Config._column.)
_SUFFIXLESS = {"vix_at_article_time", "aiq_ai_etf_intraday_gain_till_article",
               "sentiment"}

# articles.sentiment is an ordinal 1..5 label, so target_kind="sentiment" turns
# the run into 5-way classification: the head emits 5 logits and the loss is
# cross-entropy against the one-hot label.
SENTIMENT_CLASSES = 5
ACT_CLASSES = 2

CHOICES = {
    "ticker_variant": ("primary", "first"),
    "window": ("at_article", "30m_before"),
    "target_kind": ("gain_90m_after", "intraday_gain_till", "sentiment"),
    "target_transform": ("standardize", "percent", "none"),
    "pooling": ("cls", "mean"),
    "loss_type": ("mse", "huber"),
    "train_mode": ("partial_freeze", "lora"),
    "precision": ("auto", "bf16", "fp16", "fp32"),
    "monitor": ("val_loss", "val_mae", "val_mse", "val_acc", "val_f1"),
}


@dataclass
class Config:
    # ---- data (§1) ----
    dsn: str | None = None
    input_path: str | None = None
    text_column: str = "consolidated_insights"
    ticker_variant: str = "primary"
    # "gain_90m_after"    the forward return -- the real task
    # "intraday_gain_till" the move ALREADY realized when the article landed.
    #                     A positive control: the text describes this move, so a
    #                     working pipeline must score clearly above zero on it.
    target_kind: str = "gain_90m_after"
    window: str = "at_article"
    # article_in_first_market_30m marks the 449 articles published inside the
    # first 30 market minutes. They still carry values, but their "30m before"
    # anchor is clamped to the 09:30 open, so the pre-window is shorter than
    # 30m and the target spans less than 120m. Dropped by default in the
    # 30m_before window to keep the measurement uniform.
    exclude_first_market_30m: bool = True
    # articles.is_act_from_insights -- whether the distilled insights describe an
    # actionable event. None = no filter (every row); True = only actionable
    # rows; False = only the rest. NULL in the DB means "no insights", and those
    # rows are already excluded by the text NOT NULL requirement.
    act_filter: bool | None = None
    use_floats: bool = True
    # Feed articles.sentiment (1..5) as a 5-dim ONE-HOT appended to the dense
    # input vector, alongside the 3 floats. Not standardized -- a one-hot is
    # already on the right scale and centering it would destroy its meaning.
    use_sentiment: bool = False
    # Feed articles.is_act_from_insights as a 2-dim ONE-HOT: [1,0] not
    # actionable, [0,1] actionable. Same idea as use_sentiment.
    use_act: bool = False
    exclude_long_insights: bool = True
    padding: str = "max_length"
    max_length: int = 512
    group_by: list[str] = field(default_factory=lambda: ["ticker", "et_date"])
    log_transform_floats: bool = False
    target_transform: str = "standardize"
    limit: int | None = None

    # ---- model (§2) ----
    encoder_name: str = "microsoft/deberta-v3-base"
    pooling: str = "mean"
    head_hidden_dim: int = 128
    head_dropout: float = 0.25

    # ---- loss (§3) ----
    loss_type: str = "huber"
    huber_delta: float = 1.0

    # ---- training (§4) ----
    train_mode: str = "partial_freeze"
    freeze_except_last_n: int = 3
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None
    head_lr: float = 1e-3
    encoder_lr: float = 2e-5
    phase2_min_lr: float = 5e-6
    phase1_epochs: int = 4
    unfreeze_all_phase2: bool = True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    batch_size: int = 16
    grad_accum: int = 1
    gradient_checkpointing: bool = True
    max_epochs: int = 20
    precision: str = "auto"
    seed: int = 42

    # ---- early stopping (§6) ----
    patience: int = 3
    monitor: str = "val_loss"

    # ---- k-fold (§5) ----
    use_kfold: bool = True
    k_folds: int = 5

    def __post_init__(self) -> None:
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name}={value!r} must be one of {allowed}")
        if self.padding != "max_length":
            print(f"[config] WARNING padding={self.padding!r}: the spec pins "
                  f"'max_length' so batch memory is constant (§1, §10).", flush=True)
        if self.window == "30m_before" and self.ticker_variant == "primary":
            # Only the _first flavour of these columns was ever computed.
            print("[config] window=30m_before is first_mentioned_ticker only; "
                  "setting ticker_variant=first.", flush=True)
            self.ticker_variant = "first"
        cls_monitors, reg_monitors = {"val_acc", "val_f1"}, {"val_mae", "val_mse"}
        bad = cls_monitors if self.task == "regression" else reg_monitors
        if self.monitor in bad:
            raise ValueError(f"monitor={self.monitor!r} does not exist for a "
                             f"{self.task} run; use val_loss"
                             + (" / val_acc / val_f1" if self.task == "classification"
                                else " / val_mae / val_mse"))
        if self.use_sentiment and self.target_kind == "sentiment":
            raise ValueError("use_sentiment feeds articles.sentiment as an input, but "
                             "target_kind='sentiment' predicts it -- that is the label "
                             "itself. Pick one.")
        if self.use_floats and self.float_2_column == self.target_column:
            # float_2 IS intraday_gain_till_article, so feeding the floats while
            # predicting it hands the model the answer. Refuse rather than
            # report a spectacular R2 that means nothing.
            raise ValueError(
                f"target_kind={self.target_kind!r} makes the target "
                f"({self.target_column}) identical to float_2 -- that is direct "
                f"leakage. Pass use_floats=false (--no-floats) for this target.")
        if self.train_mode == "lora" and self.unfreeze_all_phase2:
            # Not an error -- phase 2 is simply not part of the LoRA path (§4).
            self.unfreeze_all_phase2 = False

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Defaults <- YAML at `path` (default.yaml if None) <- non-None overrides."""
        import yaml

        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
        elif path != DEFAULT_CONFIG_PATH:
            raise FileNotFoundError(path)

        known = {f.name: f for f in fields(cls)}
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"{path}: unknown config key(s) {sorted(unknown)}")

        merged = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**{k: _coerce(known[k], v) for k, v in merged.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Columns the ticker_variant selects (§1). Never mix the two pairs.
    def _column(self, role: str) -> str:
        """The DB column for a role, given the window and ticker flavour."""
        name = WINDOW_COLUMNS[self.window][role]
        if (self.window != "at_article" or self.ticker_variant == "primary"
                or name in _SUFFIXLESS):
            return name          # 30m_before names are already fully qualified
        return name + "_first"

    @property
    def task(self) -> str:
        return "classification" if self.target_kind == "sentiment" else "regression"

    @property
    def n_dense_features(self) -> int:
        """Width of the non-text vector concatenated to the pooled encoder output."""
        return ((3 if self.use_floats else 0)
                + (SENTIMENT_CLASSES if self.use_sentiment else 0)
                + (ACT_CLASSES if self.use_act else 0))

    @property
    def n_outputs(self) -> int:
        """Head width: 5 one-hot logits for sentiment, 1 scalar for a return."""
        return SENTIMENT_CLASSES if self.task == "classification" else 1

    @property
    def float_1_column(self) -> str:
        return self._column("float_1")

    @property
    def float_2_column(self) -> str:
        return self._column("float_2")

    @property
    def float_3_column(self) -> str:
        return self._column("float_3")

    @property
    def float_columns(self) -> tuple[str, str, str]:
        return self.float_1_column, self.float_2_column, self.float_3_column

    @property
    def target_column(self) -> str:
        return self._column(self.target_kind)


def _coerce(f: Any, value: Any) -> Any:
    """Cast `value` to the dataclass field's annotated type.

    Only the shapes this config actually uses: optional scalars, lists of str,
    and plain int/float/bool/str. A YAML `null` stays None.
    """
    if value is None:
        return None
    ann = str(f.type)
    if "list" in ann:
        return list(value)
    if "bool" in ann:
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    if "int" in ann and "str" not in ann:
        return int(value)
    if "float" in ann:
        return float(value)
    return value
