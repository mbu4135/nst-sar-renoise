"""
모든 정규화 파라미터의 단일 진실 공급원 (SSOT).

규칙
----
image 영상  : log10(linear) → minmax 정규화 → [0, 1]
ratio 영상  : log10(ratio) → minmax_symmetric 정규화 → [0, 1], 변화 없음 = 0.5

ratio normalize_symmetric(x_log) = (x_log + L) / (2L)
    x_log = 0 (== 변화 없음, linear ratio = 1) → 0.5
    x_log = +L → 1.0
    x_log = −L → 0.0
clamp [0, 1] 이후 학습/추론에 사용.

L 값 산정 (training_lareunion_ratio, 20 dates sample, 2026-05-22)
----
        p0.1      p1     p50     p99   p99.9   →  chosen L
C11  : -3.17   -2.14   -0.16   +0.69   +0.89        2.5
C22  : -3.45   -2.41   -0.19   +0.74   +0.95        2.9
C12m : -2.20   -1.53   -0.13   +0.63   +0.83        1.5

VV/VH는 patch 단위 랜덤 선택으로 같이 학습되므로, 정규화 후 분포가
일치하도록 L 을 따로 잡았다.  L_vh = 2.9 일 때 VV/VH norm 의
percentile 차이 ≤ 0.015, mean/std 거의 동일 (VV 0.447/0.116,
VH 0.445/0.114).

선택 기준: 상단 꼬리(+0.7~+1.0)를 충분히 담으면서 하단 클립을 ~1% 이하로 유지.
"""
from __future__ import annotations

import numpy as np
import torch

# Ocean / masked pixels = 0 in linear. log10(0) is −inf — replace with 1.0 (log = 0,
# norm = 0.5, neutral) and rely on the mask returned by mask_linear_zero.
_OCEAN_FILL_LINEAR = 1.0


# image norm = log10(Re²+Im²) intensity 범위. filtering INTENSITY_NORM 과 통일
# (vv=VV, vh=VH, cross=C12_mag). diffusion image stage 전용 — I2I ratio 경로 미사용.
NORM_CONFIG = {
    # VV intensity ratio (= C11.img in /LR/ratio/)
    "vv": {
        "image": {
            "mode": "minmax",
            "log_min": -12.0,
            "log_max":  10.0,
        },
        "ratio": {
            "mode": "minmax_symmetric",
            "log_abs_max": 2.5,
        },
    },
    # VH intensity ratio (= C22.img)
    "vh": {
        "image": {
            "mode": "minmax",
            "log_min": -14.0,
            "log_max":   8.0,
        },
        "ratio": {
            "mode": "minmax_symmetric",
            "log_abs_max": 2.9,
        },
    },
    # cross-pol magnitude ratio (= C12_mag.img)
    "cross": {
        "image": {
            "mode": "minmax",
            "log_min": -8.3,
            "log_max":  2.2,
        },
        "ratio": {
            "mode": "minmax_symmetric",
            "log_abs_max": 1.5,
        },
    },
}

# File-name → pol key (so call sites can be channel-explicit)
CHANNEL_TO_POL = {
    "C11":     "vv",
    "C22":     "vh",
    "C12_mag": "cross",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _is_torch(x):
    return isinstance(x, torch.Tensor)


def mask_linear_zero(x_linear):
    """True where x > 0 (valid SAR pixel). Ocean / mask is x == 0."""
    if _is_torch(x_linear):
        return x_linear > 0
    return x_linear > 0


def to_log_ratio(x_linear):
    """linear ratio → log10 ratio. Ocean pixels (x == 0) → 0 (neutral)."""
    if _is_torch(x_linear):
        safe = torch.where(x_linear > 0, x_linear,
                           torch.full_like(x_linear, _OCEAN_FILL_LINEAR))
        return torch.log10(safe)
    safe = np.where(x_linear > 0, x_linear, _OCEAN_FILL_LINEAR)
    return np.log10(safe)


def from_log_ratio(x_log):
    """log10 ratio → linear."""
    if _is_torch(x_log):
        return torch.pow(torch.tensor(10.0, dtype=x_log.dtype, device=x_log.device), x_log)
    return np.power(10.0, x_log)


# ── ratio (minmax_symmetric) ───────────────────────────────────────────────

def _ratio_L(pol: str) -> float:
    cfg = NORM_CONFIG[pol]["ratio"]
    assert cfg["mode"] == "minmax_symmetric"
    return float(cfg["log_abs_max"])


def normalize_ratio(x_log, pol: str = "vv", clip: bool = True):
    """log10 ratio → [0, 1] (0.5 = no change)."""
    L = _ratio_L(pol)
    y = (x_log + L) / (2.0 * L)
    if clip:
        if _is_torch(y):
            y = torch.clamp(y, 0.0, 1.0)
        else:
            y = np.clip(y, 0.0, 1.0)
    return y


def denormalize_ratio(x_norm, pol: str = "vv"):
    """[0, 1] norm → log10 ratio."""
    L = _ratio_L(pol)
    return x_norm * (2.0 * L) - L


# ── full pipeline (linear ↔ normalized) ────────────────────────────────────

def linear_to_norm_ratio(x_linear, pol: str = "vv", clip: bool = True):
    """학습 입력 변환 : linear ratio → log10 → normalize [0, 1].

    Ocean(x == 0) 픽셀은 norm = 0.5 가 되어 학습에 영향 없음. 별도 mask 가
    필요하면 mask_linear_zero(x_linear) 를 같이 가져가서 loss 단계에서 사용.
    """
    return normalize_ratio(to_log_ratio(x_linear), pol=pol, clip=clip)


def norm_to_linear_ratio(x_norm, pol: str = "vv"):
    """추론 출력 변환 : norm [0, 1] → log10 → 10**x → linear ratio."""
    return from_log_ratio(denormalize_ratio(x_norm, pol=pol))


# ── image (legacy, kept for diffusion stage) ────────────────────────────────

def normalize_image(x_log, pol: str = "vv"):
    cfg = NORM_CONFIG[pol].get("image")
    if cfg is None:
        raise KeyError(f"no image normalization configured for pol={pol}")
    if cfg["mode"] == "minmax":
        return (x_log - cfg["log_min"]) / (cfg["log_max"] - cfg["log_min"])
    raise NotImplementedError(f"unknown image norm mode: {cfg['mode']}")


def denormalize_image(x_norm, pol: str = "vv"):
    cfg = NORM_CONFIG[pol].get("image")
    if cfg is None:
        raise KeyError(f"no image normalization configured for pol={pol}")
    if cfg["mode"] == "minmax":
        return x_norm * (cfg["log_max"] - cfg["log_min"]) + cfg["log_min"]
    raise NotImplementedError(f"unknown image norm mode: {cfg['mode']}")


if __name__ == "__main__":
    print("=== ratio normalization (minmax_symmetric, 0.5 = no change) ===")
    for pol, cfg in NORM_CONFIG.items():
        r = cfg.get("ratio")
        if r is None:
            continue
        L = r["log_abs_max"]
        print(f"  pol={pol:<6s}  L={L:.2f}  log10 ∈ [{-L:+.2f}, {+L:+.2f}]  "
              f"linear ∈ [{10**-L:.4f}, {10**L:.1f}]")
    print("\nfile → pol mapping:", CHANNEL_TO_POL)
