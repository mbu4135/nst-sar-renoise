"""
DDPM linear beta schedule + 사전 계산된 텐서.

β_1 = 1e-4, β_T = 0.02, T = 1000 (default, RNSD 와 동일).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDPMSchedule:
    """
    사전 계산값 보관.

    fields (all on `device`, dtype float32, length T):
        betas
        alphas
        alpha_bar
        sqrt_alpha_bar
        sqrt_one_minus_alpha_bar
    """
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor
    sqrt_alpha_bar: torch.Tensor
    sqrt_one_minus_alpha_bar: torch.Tensor

    def to(self, device: str | torch.device) -> "DDPMSchedule":
        return DDPMSchedule(
            T=self.T,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bar=self.alpha_bar.to(device),
            sqrt_alpha_bar=self.sqrt_alpha_bar.to(device),
            sqrt_one_minus_alpha_bar=self.sqrt_one_minus_alpha_bar.to(device),
        )


def make_linear_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DDPMSchedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return DDPMSchedule(
        T=T,
        betas=betas.to(device),
        alphas=alphas.to(device),
        alpha_bar=alpha_bar.to(device),
        sqrt_alpha_bar=torch.sqrt(alpha_bar).to(device),
        sqrt_one_minus_alpha_bar=torch.sqrt(1.0 - alpha_bar).to(device),
    )
