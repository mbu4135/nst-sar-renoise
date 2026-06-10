"""
cs (ratio 영상) 생성 파이프라인.

raw_ratio  = y / (s + eps)              # linear, 변화 없음 ≈ 1
log_ratio  = log10(raw_ratio + eps)     # 변화 없음 ≈ 0
norm_ratio = normalize_ratio(log_ratio) # [0, 1], 변화 없음 = 0.5
cs         = ratio_denoiser(norm_ratio) # pretrained, freeze
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from nstsr.config.norm_config import normalize_ratio


def _build_ratio_norm(y, s, eps: float, pol: str):
    """공통: y, s → norm_ratio (numpy or tensor in-place 호환)."""
    raw   = y / (s + eps)
    if isinstance(raw, torch.Tensor):
        lograt = torch.log10(raw + eps)
    else:
        lograt = np.log10(raw + eps)
    return normalize_ratio(lograt, pol=pol)


@torch.no_grad()
def build_ratio_cs(
    y_linear: torch.Tensor,
    s_linear: torch.Tensor,
    ratio_denoiser: Optional[torch.nn.Module] = None,
    eps: float = 1e-8,
    pol: str = "vv",
) -> torch.Tensor:
    """
    Parameters
    ----------
    y_linear, s_linear : [B, 1, H, W] or [1, H, W] linear-domain SAR.
    ratio_denoiser     : pretrained nn.Module (eval/freeze). None 이면 norm_ratio 를 그대로 반환.

    Returns
    -------
    cs : [B, 1, H, W] in [0, 1].
    """
    norm_ratio = _build_ratio_norm(y_linear, s_linear, eps=eps, pol=pol)
    if ratio_denoiser is None:
        return norm_ratio
    if norm_ratio.ndim == 3:
        norm_ratio = norm_ratio.unsqueeze(0)
    ratio_denoiser.eval()
    cs = ratio_denoiser(norm_ratio)
    return cs


def build_ratio_cs_numpy(
    y_linear: np.ndarray,
    s_linear: np.ndarray,
    ratio_denoiser: Optional[torch.nn.Module] = None,
    eps: float = 1e-8,
    pol: str = "vv",
    device: str = "cpu",
) -> np.ndarray:
    """prepare_data 단계용 numpy in / numpy out 래퍼 (single forward)."""
    y_t = torch.from_numpy(np.ascontiguousarray(y_linear)).float()
    s_t = torch.from_numpy(np.ascontiguousarray(s_linear)).float()
    if y_t.ndim == 2:
        y_t = y_t.unsqueeze(0).unsqueeze(0)
    if s_t.ndim == 2:
        s_t = s_t.unsqueeze(0).unsqueeze(0)
    if ratio_denoiser is not None:
        ratio_denoiser = ratio_denoiser.to(device).eval()
        y_t, s_t = y_t.to(device), s_t.to(device)
    cs = build_ratio_cs(y_t, s_t, ratio_denoiser=ratio_denoiser, eps=eps, pol=pol)
    return cs.squeeze().detach().cpu().numpy()


def build_ratio_cs_multilook_numpy(
    y_linear: np.ndarray,
    s_linear: np.ndarray,
    looks: int = 16,
    eps: float = 1e-8,
    pol: str = "vv",
) -> np.ndarray:
    """denoiser 없이 고전적 multilook 으로 cs 생성.

    linear ratio = y/s 를 비중첩 looks×looks 블록 평균(= multilook) → log10 →
    multilook 전용 L(ratio_ml)로 symmetric min-max 정규화.

    1. raw = y / (s + eps)                       (linear, 변화 없음 ≈ 1)
    2. ocean(y==0 | s==0) 픽셀은 평균에서 제외 (masked block mean)
    3. looks×looks 블록 평균 → 해상도 1/looks 로 축소
    4. norm = normalize_ratio(log10(blk + eps), ml=True)   ([0,1], 무변화 0.5)
    5. 완전 ocean 블록 → 0 (기존 cs[~mask]=0 convention 과 일치)

    Returns cs : [H//looks, W//looks] float32 in [0, 1].
    """
    y = np.asarray(y_linear, dtype=np.float32)
    s = np.asarray(s_linear, dtype=np.float32)
    assert y.shape == s.shape and y.ndim == 2, f"expect 2D same-shape, got {y.shape}/{s.shape}"
    H, W = y.shape
    b = int(looks)
    Hb, Wb = H // b, W // b
    if Hb < 1 or Wb < 1:
        raise ValueError(f"image {H}x{W} too small for {b}x{b} multilook")

    mask = ((y > 0) & (s > 0)).astype(np.float32)
    raw = (y / (s + eps)) * mask                       # ocean → 0, 평균에서 제외
    raw = raw[:Hb * b, :Wb * b].reshape(Hb, b, Wb, b)
    msk = mask[:Hb * b, :Wb * b].reshape(Hb, b, Wb, b)
    num = raw.sum(axis=(1, 3))
    den = msk.sum(axis=(1, 3))                          # 블록당 valid look 수
    valid = den > 0
    blk = np.where(valid, num / np.where(valid, den, 1.0), 0.0).astype(np.float32)

    norm = normalize_ratio(np.log10(blk + eps), pol=pol, ml=True).astype(np.float32)
    norm[~valid] = 0.0
    return norm


def _patch_offsets(total: int, patch: int, stride: int):
    offs = list(range(0, total - patch + 1, stride))
    if not offs or offs[-1] != total - patch:
        offs.append(total - patch)
    return offs


@torch.no_grad()
def build_ratio_cs_patched_numpy(
    y_linear: np.ndarray,
    s_linear: np.ndarray,
    ratio_denoiser: torch.nn.Module,
    patch_size: int = 512,
    stride: int = 256,
    eps: float = 1e-8,
    pol: str = "vv",
    device: str = "cpu",
) -> np.ndarray:
    """
    큰 영상(예: 3680×12960)용 patch-based cs 계산. Hanning overlap-tile blending.

    1. norm_ratio = normalize_ratio(log10(y/s))  (full image, [0,1])
    2. ocean( y==0 | s==0 ) → 0  (I2I ratio 모델 학습 convention 과 일치)
    3. 타일별 ratio_denoiser forward → Hanning 가중 평균
    4. ocean 다시 0 으로 강제 복원

    Returns cs : [H, W] float32 in [0, 1].
    """
    y = np.asarray(y_linear, dtype=np.float32)
    s = np.asarray(s_linear, dtype=np.float32)
    assert y.shape == s.shape and y.ndim == 2, f"expect 2D same-shape, got {y.shape}/{s.shape}"
    H, W = y.shape

    norm = _build_ratio_norm(y, s, eps=eps, pol=pol).astype(np.float32)  # [0,1] clipped
    mask = (y > 0) & (s > 0)
    norm = np.where(mask, norm, 0.0).astype(np.float32)

    p = patch_size
    while p > H or p > W:
        p //= 2
    if p < 1:
        raise ValueError(f"image {H}x{W} too small for any patch")
    xs = _patch_offsets(W, p, stride)
    ys = _patch_offsets(H, p, stride)
    hann = np.outer(np.hanning(p), np.hanning(p)).astype(np.float64)
    accum = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)

    ratio_denoiser = ratio_denoiser.to(device).eval()
    for y0 in ys:
        for x0 in xs:
            tile = norm[y0:y0 + p, x0:x0 + p]
            t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            pred = ratio_denoiser(t)[0, 0].detach().cpu().numpy()
            accum[y0:y0 + p, x0:x0 + p] += pred * hann
            cnt[y0:y0 + p, x0:x0 + p] += hann
    cnt = np.where(cnt == 0, 1, cnt)
    cs = (accum / cnt).astype(np.float32)
    cs[~mask] = 0.0
    return cs
