"""
TCCAM (Time-and-Camera-aware Conditioning Affine Module) — 현 단계 단순화 버전.

원 RNSD 는 (timestep + camera settings) → MLP → affine (γ, β) 형태이지만,
본 단계에서는 camera-settings 자리에 명시적 metadata 가 없으므로
**timestep 만** 받는다. `extra` 인자 자리만 마련해 두어,
추후 편파 one-hot 등을 합류시킬 수 있게 한다.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t : [B] long.
    return : [B, dim] float — DDPM 표준 sinusoidal positional embedding.
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """
    timestep → sinusoidal → MLP → time embedding vector.

    추후 편파 확장 시 `extra` (B, extra_dim) 를 받아 합치는 자리를 마련해 둔다.

    Output shape: [B, hidden].
    """

    def __init__(self, dim: int, hidden: int, extra_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.extra_dim = extra_dim
        in_dim = dim + extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        t     : [B] long
        extra : [B, extra_dim] or None — 추후 편파 metadata.
        """
        emb = sinusoidal_embedding(t, self.dim)
        if self.extra_dim > 0:
            if extra is None:
                extra = torch.zeros(emb.size(0), self.extra_dim, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, extra], dim=-1)
        return self.mlp(emb)
