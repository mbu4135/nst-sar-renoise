"""
SAR 영상 전처리/증강 — log10, normalize, crop, flip.

3개 입력(y, s, cs)은 random crop/flip 시 **동일 좌표/방향**으로 적용된다.
"""
from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
import torch


# ── log domain 왕복 ─────────────────────────────────────────────────────────

def to_log10(x_linear, eps: float = 1e-8):
    """log10(x + eps). torch.Tensor / numpy 모두 지원."""
    if isinstance(x_linear, torch.Tensor):
        return torch.log10(x_linear + eps)
    return np.log10(x_linear + eps)


def from_log10(x_log):
    """10 ** x."""
    if isinstance(x_log, torch.Tensor):
        return torch.pow(torch.tensor(10.0, dtype=x_log.dtype, device=x_log.device), x_log)
    return np.power(10.0, x_log)


# ── 텐서화 ──────────────────────────────────────────────────────────────────

def to_tensor_2d(arr) -> torch.Tensor:
    """numpy [H, W] / [1, H, W] → torch float32 [1, H, W]."""
    if isinstance(arr, torch.Tensor):
        t = arr.float()
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[0] != 1:
        raise ValueError(f"expected single-channel input, got shape {tuple(t.shape)}")
    return t


# ── 동기화된 random augmentation (3개 이상의 입력 동시) ───────────────────────

def random_crop(tensors: Sequence[torch.Tensor], patch: int) -> Tuple[torch.Tensor, ...]:
    """동일 좌표 random crop. tensors 는 모두 동일한 [C, H, W] shape 가정."""
    assert len(tensors) > 0
    _, h, w = tensors[0].shape
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than patch {patch}")
    top  = random.randint(0, h - patch)
    left = random.randint(0, w - patch)
    return tuple(t[:, top:top + patch, left:left + patch] for t in tensors)


def center_crop(tensors: Sequence[torch.Tensor], patch: int) -> Tuple[torch.Tensor, ...]:
    _, h, w = tensors[0].shape
    top  = (h - patch) // 2
    left = (w - patch) // 2
    return tuple(t[:, top:top + patch, left:left + patch] for t in tensors)


def random_hflip(tensors: Sequence[torch.Tensor], p: float = 0.5) -> Tuple[torch.Tensor, ...]:
    if random.random() < p:
        return tuple(torch.flip(t, dims=[-1]) for t in tensors)
    return tuple(tensors)


def random_vflip(tensors: Sequence[torch.Tensor], p: float = 0.5) -> Tuple[torch.Tensor, ...]:
    if random.random() < p:
        return tuple(torch.flip(t, dims=[-2]) for t in tensors)
    return tuple(tensors)


def augment_triplet(
    y: torch.Tensor,
    s: torch.Tensor,
    cs: torch.Tensor,
    patch: int | None = None,
    hflip: bool = True,
    vflip: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y, s, cs) 에 동기화된 crop + flip 적용."""
    triplet: Tuple[torch.Tensor, ...] = (y, s, cs)
    if patch is not None:
        triplet = random_crop(triplet, patch)
    if hflip:
        triplet = random_hflip(triplet)
    if vflip:
        triplet = random_vflip(triplet)
    return triplet  # type: ignore[return-value]
