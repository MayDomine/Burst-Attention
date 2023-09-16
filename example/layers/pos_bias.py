import math
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, seq_dim, offset):
    if x.size(seq_dim) < cos.size(seq_dim):
        cos = cos.narrow(seq_dim, offset, x.size(seq_dim))
        sin = sin.narrow(seq_dim, offset, x.size(seq_dim))
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        dtype=torch.half,
        persistent=True,
        mixed_precision=False,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        if mixed_precision:
            self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        else:
            self.register_buffer("inv_freq", inv_freq.to(self.dtype), persistent=persistent)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        self.mixed_precision = mixed_precision

        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def _update_cos_sin_tables(self, x, seq_dim, offset):
        seq_len = x.size(seq_dim) + offset
        if seq_len > self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            for i in range(x.dim() - 1):
                if i != seq_dim:
                    emb = emb.unsqueeze_(i)
            if self.mixed_precision:
                self._cos_cached = emb.cos().to(self.dtype)
                self._sin_cached = emb.sin().to(self.dtype)
            else:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim, offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_dim = (seq_dim + k.dim()) % k.dim()
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim, offset)
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dim, offset),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dim, offset),
        )