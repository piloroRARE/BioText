"""Configuration validation for BioText.

This module centralizes sanity checks for model settings to avoid invalid
training states and runtime surprises.
"""


def validate_config(cfg):
    """Validate that the configuration is internally consistent."""
    if cfg is None:
        raise ValueError("cfg cannot be None")

    if getattr(cfg, "vocab_size", None) is None or cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if getattr(cfg, "seq_len", None) is None or cfg.seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if getattr(cfg, "d_model", None) is None or cfg.d_model <= 0:
        raise ValueError("d_model must be positive")
    if getattr(cfg, "n_layers", None) is None or cfg.n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if getattr(cfg, "d_ff", None) is None or cfg.d_ff <= 0:
        raise ValueError("d_ff must be positive")
    if getattr(cfg, "n_heads", None) is None or cfg.n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    if getattr(cfg, "batch_size", None) is None or cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if getattr(cfg, "grad_accum", None) is None or cfg.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if getattr(cfg, "train_steps", None) is None or cfg.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if getattr(cfg, "lr", None) is None or cfg.lr <= 0:
        raise ValueError("lr must be positive")

    if getattr(cfg, "max_edges", None) is None or cfg.max_edges <= 0:
        raise ValueError("max_edges must be positive")
    if getattr(cfg, "min_edges", None) is None or cfg.min_edges <= 0:
        raise ValueError("min_edges must be positive")
    if cfg.max_edges < cfg.min_edges:
        raise ValueError("max_edges must be >= min_edges")

    if getattr(cfg, "sleep_every", None) is not None and cfg.sleep_every <= 0:
        raise ValueError("sleep_every must be positive")
    if getattr(cfg, "sleep_steps", None) is not None and cfg.sleep_steps <= 0:
        raise ValueError("sleep_steps must be positive")

    if getattr(cfg, "topk_ratio", None) is not None and not (0 < cfg.topk_ratio <= 1):
        raise ValueError("topk_ratio must be in (0, 1]")
    if getattr(cfg, "topk_min", None) is not None and cfg.topk_min < 0:
        raise ValueError("topk_min must be >= 0")
    if getattr(cfg, "topk_min", None) is not None and cfg.topk_min > cfg.d_model:
        raise ValueError("topk_min cannot exceed d_model")

    if getattr(cfg, "gen_temperature", None) is not None and cfg.gen_temperature <= 0:
        raise ValueError("gen_temperature must be positive")
    if getattr(cfg, "gen_topk", None) is not None and cfg.gen_topk < 0:
        raise ValueError("gen_topk must be >= 0")
    if getattr(cfg, "gen_rep_penalty", None) is not None and cfg.gen_rep_penalty <= 0:
        raise ValueError("gen_rep_penalty must be positive")

    return True
