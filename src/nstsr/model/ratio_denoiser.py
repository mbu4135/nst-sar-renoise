"""
Pretrained ratio denoiser wrapper.

cs = ratio_denoiser(norm_ratio) 단계에서 쓰는 사전학습 ratio denoise 모델을 로드한다.
(b, 1, H, W) 정규화 ratio [0,1] → (b, 1, H, W) denoised ratio [0,1].
학습/추론 중에는 항상 eval + freeze.

arch
----
"identity"     : placeholder (입력 그대로 반환). 파이프라인 시동/sanity 용.
"i2i_unetsar"  : filtering 프로젝트의 I2I ratio 모델 (UNetSAR, base_ch=32).
                 nst-sar-filtering/checkpoints/i2i_ratio_C11_ft/best_model.pth 와 호환.
                 norm 은 renoise pol="vv" (L=2.5) == filtering C11 (L=2.5) 로 통일됨.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class IdentityRatioDenoiser(nn.Module):
    """placeholder: 입력 norm_ratio 를 그대로 반환 (sanity-check / 파이프라인 시동용)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _extract_state_dict(state):
    """다양한 체크포인트 포맷에서 state_dict 추출."""
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def build_ratio_denoiser(
    arch: str = "identity",
    ckpt_path: Optional[str | Path] = None,
    device: str = "cuda",
    base_ch: int = 32,
    strict: bool = True,
) -> nn.Module:
    """
    Parameters
    ----------
    arch       : "identity" | "i2i_unetsar".
    ckpt_path  : 가중치 파일 경로. None 이면 random/identity.
    device     : "cuda" / "cpu".
    base_ch    : i2i_unetsar 의 UNet 폭. i2i_ratio_C11_ft 는 32.
    strict     : load_state_dict strict.

    Returns
    -------
    nn.Module — eval mode, requires_grad=False.
    """
    if arch == "identity":
        model: nn.Module = IdentityRatioDenoiser()
    elif arch in ("i2i_unetsar", "unet_sar"):
        from nstsr.model.unet_sar import UNetSAR
        model = UNetSAR(base_ch=base_ch, in_ch=1, out_ch=1)
    else:
        raise NotImplementedError(f"unknown ratio denoiser arch: {arch}")

    if ckpt_path is not None:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(_extract_state_dict(state), strict=strict)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
