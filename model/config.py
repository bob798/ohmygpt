from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 16
    n_kv_heads: int = 8
    max_seq_len: int = 512
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # FFN hidden dim; if None, computed as multiple_of-rounded 8/3 * dim
    ffn_hidden: int | None = None
    multiple_of: int = 64

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.ffn_hidden is None:
            hidden = int(8 / 3 * self.dim)
            self.ffn_hidden = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def small_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(dim=256, n_layers=4, n_heads=8, n_kv_heads=4)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def base_config(**overrides) -> ModelConfig:
    cfg = ModelConfig()  # defaults are the base preset
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
