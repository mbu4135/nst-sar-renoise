"""
Algorithm 1 (DDPM training step) — RNSD 본문과 동일.

x0 = y  (single-look noisy SAR, log-normalized)
t  ~ U(0, T)
ε  ~ N(0, I)
x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
loss = MSE(model(x_t, t, s, cs), ε)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nstsr.diffusion.schedule import DDPMSchedule


def diffuse(x0: torch.Tensor, t: torch.Tensor, schedule: DDPMSchedule, noise: torch.Tensor | None = None):
    """
    Forward q(x_t | x_0).

    x0    : [B, C, H, W]
    t     : [B] long, ∈ [0, T)
    return: (x_t, noise)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab   = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    x_t = sqrt_ab * x0 + sqrt_1_ab * noise
    return x_t, noise


def training_step(batch, model, schedule: DDPMSchedule, device: str = "cuda") -> torch.Tensor:
    """
    한 batch loss 반환.

    batch : dict with keys y, s, cs — 모두 [B, 1, H, W], log-normalized [0, 1].
    model : conditional UNet (forward(x_t, t, s, cs)).
    """
    y  = batch["y"].to(device, non_blocking=True)
    s  = batch["s"].to(device, non_blocking=True)
    cs = batch["cs"].to(device, non_blocking=True)

    B = y.size(0)
    t = torch.randint(0, schedule.T, (B,), device=device)
    x_t, eps = diffuse(y, t, schedule)

    eps_hat = model(x_t, t, s, cs)
    loss = F.mse_loss(eps_hat, eps)
    return loss
