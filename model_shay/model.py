"""Architecture, loss factory and checkpointing (spec §2, §3, §4).

    text_ids/attention_mask -> encoder -> pooled (cls | mean)
    float_1..3 (standardized) -> concat directly, no MLP at this size
    concat -> Linear(hidden+3, 128) -> GELU -> Dropout -> Linear(128, 1)

The two trainable modes differ only in the freeze/adapter step, never in the
loop: `partial_freeze` unfreezes the last N encoder layers, `lora` injects
adapters on the attention query/value projections and freezes the base.

Checkpoints are a single .pt carrying weights, the fitted float scaler, the
target transform and the resolved config -- everything `predict` needs to
rebuild the model with no YAML and no peft. LoRA runs are merged into the base
weights before saving, so both modes produce the same artifact shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model_shay.config import Config
from model_shay.data import FloatScaler, TargetTransform

CHECKPOINT_FORMAT = 1
N_FLOATS = 3


class TextFloatsRegressor(nn.Module):
    def __init__(self, cfg: Config, *, pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.pooling = cfg.pooling
        if pretrained:
            encoder = AutoModel.from_pretrained(cfg.encoder_name)
        else:
            # Loading a checkpoint: the weights are about to be overwritten, so
            # skip fetching pretrained ones and build from the architecture
            # config alone. Keeps `predict` off the network and much faster.
            encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg.encoder_name))
        # .float() is NOT redundant. deberta-v3-base's config declares
        # torch_dtype: float16, and transformers >=5 defaults dtype="auto",
        # so it loads fp16 PARAMETERS -- which train to NaN on the first step
        # (AdamW moments and LayerNorm in fp16 overflow) and mismatch the fp32
        # head outside autocast. Master weights stay fp32 here; mixed precision
        # is the autocast context's job in train.py, not the weights'.
        self.encoder = encoder.float()
        # Text-only runs drop the float inputs entirely rather than zeroing
        # them, so the head has no dead weights to regularize around.
        # floats (3) and/or the sentiment one-hot (5), concatenated to the pooled text
        self.n_features = cfg.n_dense_features
        self.n_outputs = cfg.n_outputs
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden + self.n_features, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.n_outputs),
        )

    def pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_state[:, 0]
        # Mask-weighted mean over real tokens only -- with flat 512 padding the
        # pad positions are a large majority, so the mask is not optional.
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                floats: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        if self.n_features:
            pooled = torch.cat([pooled, floats.to(pooled.dtype)], dim=-1)
        out = self.head(pooled)
        # (B, 5) logits for classification; (B,) for a scalar regression target.
        return out if self.n_outputs > 1 else out.squeeze(-1)


# --------------------------------------------------------------------------- #
# Loss (§3): one factory, so switching costs no other code change.
# --------------------------------------------------------------------------- #
def build_loss(cfg: Config) -> nn.Module:
    if cfg.task == "classification":
        # One-hot target == integer class index under CrossEntropyLoss, which
        # applies log_softmax internally -- so the head stays raw logits.
        return nn.CrossEntropyLoss()
    if cfg.loss_type == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=cfg.huber_delta)


# --------------------------------------------------------------------------- #
# Freezing / adapters (§4)
# --------------------------------------------------------------------------- #
def encoder_layers(encoder) -> nn.ModuleList:
    """The transformer block list, wherever this architecture keeps it."""
    for path in ("encoder.layer", "encoder.layers", "layers", "transformer.layer"):
        obj: Any = encoder
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError(f"cannot locate encoder layers on {type(encoder).__name__}")


def apply_partial_freeze(model: TextFloatsRegressor, last_n: int) -> None:
    """Freeze the encoder except its last N blocks (+ pooler, if any)."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    layers = encoder_layers(model.encoder)
    last_n = max(0, min(last_n, len(layers)))
    for layer in layers[len(layers) - last_n:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    # DeBERTa's AutoModel has no pooler; BERT-likes do. Unfreeze it when present.
    pooler = getattr(model.encoder, "pooler", None)
    if pooler is not None:
        for p in pooler.parameters():
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] last {last_n}/{len(layers)} encoder block(s) + head trainable "
          f"({trainable/1e6:.1f}M params)", flush=True)


def detect_lora_targets(encoder) -> list[str]:
    """Attention query/value projection names for this architecture."""
    names = {n.rsplit(".", 1)[-1] for n, _ in encoder.named_modules()}
    for candidate in (["query_proj", "value_proj"], ["query", "value"], ["q_proj", "v_proj"]):
        if set(candidate) <= names:
            return candidate
    raise RuntimeError(f"cannot autodetect LoRA targets; set lora_target_modules. "
                       f"Saw module names like {sorted(names)[:12]}")


def apply_lora(model: TextFloatsRegressor, cfg: Config) -> None:
    """Adapters on attention q/v; base weights frozen, head stays trainable."""
    from peft import LoraConfig, get_peft_model

    targets = cfg.lora_target_modules or detect_lora_targets(model.encoder)
    model.encoder = get_peft_model(model.encoder, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", target_modules=list(targets)))
    for p in model.head.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] r={cfg.lora_r} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout} "
          f"on {targets} ({trainable/1e6:.2f}M trainable params)", flush=True)


def prepare(model: TextFloatsRegressor, cfg: Config) -> None:
    """The one step that differs between train modes."""
    if cfg.train_mode == "lora":
        apply_lora(model, cfg)
    else:
        apply_partial_freeze(model, cfg.freeze_except_last_n)


def head_and_encoder_groups(model: TextFloatsRegressor, cfg: Config) -> list[dict]:
    """Phase 1 / LoRA: head at head_lr, everything else trainable at encoder_lr."""
    head = [p for p in model.head.parameters() if p.requires_grad]
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": cfg.head_lr}]
    if enc:
        groups.append({"params": enc, "lr": cfg.encoder_lr})
    return groups


def discriminative_groups(model: TextFloatsRegressor, cfg: Config) -> list[dict]:
    """Phase 2: unfreeze everything, LR ramped phase2_min_lr -> encoder_lr by depth."""
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    layers = encoder_layers(model.encoder)
    n = len(layers)
    layer_ids = {id(p) for layer in layers for p in layer.parameters()}

    groups = [{"params": list(model.head.parameters()), "lr": cfg.head_lr}]
    for i, layer in enumerate(layers):
        frac = i / max(n - 1, 1)
        lr = cfg.phase2_min_lr + frac * (cfg.encoder_lr - cfg.phase2_min_lr)
        groups.append({"params": list(layer.parameters()), "lr": lr})
    # Embeddings and any loose encoder tensors (DeBERTa's rel_embeddings) get
    # the floor -- they are the earliest thing in the stack.
    rest = [p for p in model.encoder.parameters() if id(p) not in layer_ids]
    if rest:
        groups.append({"params": rest, "lr": cfg.phase2_min_lr})
    print(f"[phase2] full encoder unfrozen, LR {cfg.phase2_min_lr:.1e} -> "
          f"{cfg.encoder_lr:.1e} across {n} layer(s)", flush=True)
    return groups


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model: TextFloatsRegressor, cfg: Config,
                    scaler: FloatScaler, target_tf: TargetTransform,
                    metrics: dict | None = None, extra: dict | None = None) -> Path:
    """Write the single .pt `predict` runs against.

    A LoRA encoder is merged into its base weights first, so the saved
    state_dict is architecturally identical to a partial_freeze one and loading
    never needs peft installed.
    """
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg.as_dict(),
        "float_scaler": {"mean": scaler.mean, "scale": scaler.scale,
                         "log_transform": scaler.log_transform},
        "target_transform": {"kind": target_tf.kind, "mean": target_tf.mean,
                             "std": target_tf.std},
        "metrics": metrics or {},
        **(extra or {}),
    }, path)
    print(f"[checkpoint] {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[TextFloatsRegressor, Config, FloatScaler, TargetTransform, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {ckpt.get('format_version')} "
                         f"!= expected {CHECKPOINT_FORMAT}")
    cfg = Config(**ckpt["config"])
    model = TextFloatsRegressor(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    scaler = FloatScaler(**ckpt["float_scaler"])
    target_tf = TargetTransform(**ckpt["target_transform"])
    return model, cfg, scaler, target_tf, ckpt.get("metrics", {})
