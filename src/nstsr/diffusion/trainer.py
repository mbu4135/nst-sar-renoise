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
    한 batch masked ε-MSE loss 반환 (renoise speckle 설계).

    batch : dict with
        r     : [B, 1, H, W]  normalized target r = norm(log10(y/μ))  (= x0).
        cond  : [B, C, H, W]  conditioning (μ, D_A, shadow) concat.
        cmask : [B, 1, H, W]  loss mask ∈ {0,1} (shadow/change/no-data 제외).
    model : conditional UNet (forward(x_t, t, cond)).

    shadow·change·no-data 픽셀은 cmask=0 으로 loss 에서 제외 (유효 speckle 타깃이 없음).
    """
    r     = batch["r"].to(device, non_blocking=True)
    cond  = batch["cond"].to(device, non_blocking=True)
    cmask = batch["cmask"].to(device, non_blocking=True)

    B = r.size(0)
    t = torch.randint(0, schedule.T, (B,), device=device)
    x_t, eps = diffuse(r, t, schedule)

    eps_hat = model(x_t, t, cond)
    se = (eps_hat - eps) ** 2 * cmask          # per-pixel masked squared error
    loss = se.sum() / cmask.sum().clamp_min(1.0)
    return loss
