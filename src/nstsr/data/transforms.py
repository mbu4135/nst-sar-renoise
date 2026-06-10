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


# ── cs(저해상 multilook) 정렬 crop ──────────────────────────────────────────
# cs 는 y/s 보다 ratio 배 작게 저장(=looks). upsample 없이 같은 영역을 자르려면
# y/s 의 crop offset 을 ratio 의 배수로 두고 cs 는 offset/ratio·patch/ratio 로 자른다.

def _cs_ratio(y: torch.Tensor, cs: torch.Tensor) -> int:
    """y 대비 cs 의 다운샘플 배수 (정수). cs 가 같은 크기면 1."""
    rh = y.shape[-2] // cs.shape[-2]
    rw = y.shape[-1] // cs.shape[-1]
    if rh != rw or rh * cs.shape[-2] != y.shape[-2] or rw * cs.shape[-1] != y.shape[-1]:
        raise ValueError(f"cs ratio 정수 아님: y={tuple(y.shape)} cs={tuple(cs.shape)}")
    return rh


def _crop_triplet(y, s, cs, patch, top, left, r):
    yc = y[:, top:top + patch, left:left + patch]
    sc = s[:, top:top + patch, left:left + patch]
    if r == 1:
        cc = cs[:, top:top + patch, left:left + patch]
    else:
        pc = patch // r
        cc = cs[:, top // r: top // r + pc, left // r: left // r + pc]
    return yc, sc, cc


def random_crop_triplet(y, s, cs, patch):
    """y/s(고해상) + cs(저해상) 정렬 random crop. offset 은 ratio 배수."""
    _, h, w = y.shape
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than patch {patch}")
    r = _cs_ratio(y, cs)
    if patch % r:
        raise ValueError(f"patch {patch} 가 cs ratio {r} 로 나눠떨어지지 않음")
    top  = random.randint(0, (h - patch) // r) * r
    left = random.randint(0, (w - patch) // r) * r
    return _crop_triplet(y, s, cs, patch, top, left, r)


def center_crop_triplet(y, s, cs, patch):
    """y/s + cs 정렬 center crop (val 용)."""
    _, h, w = y.shape
    r = _cs_ratio(y, cs)
    top  = ((h - patch) // 2 // r) * r
    left = ((w - patch) // 2 // r) * r
    return _crop_triplet(y, s, cs, patch, top, left, r)


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
    """(y, s, cs) 에 동기화된 crop + flip 적용.

    cs 가 y/s 보다 저해상(저장 multilook)이면 ratio 배수 offset 으로 정렬 crop —
    upsample 없이 같은 영역. flip 은 해상도 무관하게 셋 모두 적용.
    """
    if patch is not None:
        y, s, cs = random_crop_triplet(y, s, cs, patch)
    triplet: Tuple[torch.Tensor, ...] = (y, s, cs)
    if hflip:
        triplet = random_hflip(triplet)
    if vflip:
        triplet = random_vflip(triplet)
    return triplet  # type: ignore[return-value]
