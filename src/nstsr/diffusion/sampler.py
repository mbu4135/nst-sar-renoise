"""
DIPS-Basic sampler (Algorithm 2 의 단순 버전).

지수 스케줄로 T 개 step 을 S 개 step 으로 줄여 DDIM-style deterministic update.
DIPS-Advanced (one-step distillation) 은 후속 단계 — TODO.
"""
from __future__ import annotations

import math
from typing import List

import torch

from nstsr.diffusion.schedule import DDPMSchedule


def dips_basic_schedule(T: int = 1000, S: int = 30, t_last: int = 4, r: float = 10.0) -> List[int]:
    """
    t_i = t_last + (T - t_last) * (exp(r * (i-1)/(S-1)) - 1) / (exp(r) - 1)

    return : [t_S, ..., t_1, 0]  — 큰 t 부터 작은 t, 마지막에 0.
    """
    denom = math.exp(r) - 1.0
    ts = []
    for i in range(1, S + 1):
        frac = (math.exp(r * (i - 1) / (S - 1)) - 1.0) / denom
        ti = int(round(t_last + (T - t_last - 1) * frac))
        ts.append(max(0, min(T - 1, ti)))
    ts = sorted(set(ts), reverse=True)
    if ts[-1] != 0:
        ts.append(0)
    return ts


@torch.no_grad()
def sample(
    model,
    schedule: DDPMSchedule,
    cond: torch.Tensor,
    shape: tuple | None = None,
    device: str = "cuda",
    S: int = 30,
    t_last: int = 4,
    r: float = 10.0,
    eta: float = 0.0,
    clamp_x0: bool = True,
) -> torch.Tensor:
    """
    DDIM-style sampling (eta>0 면 stochastic — speckle 텍스처 위해 권장).

    cond  : [B, C, H, W] conditioning (μ, D_A, shadow) concat.
    shape : 생성할 r 의 shape (B,1,H,W). 기본은 cond 로부터.
    return: x_0 = 생성된 normalized r (∈ [-1,1] 근처). 추론에서 ŷ=μ·10**denormalize_r(r̂).
    """
    if shape is None:
        shape = (cond.shape[0], 1, cond.shape[2], cond.shape[3])
    model.eval()
    schedule = schedule.to(device)

    x_t = torch.randn(shape, device=device)
    ts = dips_basic_schedule(T=schedule.T, S=S, t_last=t_last, r=r)

    for i in range(len(ts) - 1):
        t      = ts[i]
        t_next = ts[i + 1]
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_hat  = model(x_t, t_tensor, cond)

        ab_t    = schedule.alpha_bar[t]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        x0_pred = (x_t - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        if clamp_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        if eta == 0.0:
            dir_xt = torch.sqrt(1.0 - ab_next) * eps_hat
            x_t = torch.sqrt(ab_next) * x0_pred + dir_xt
        else:
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_next)
            dir_xt = torch.sqrt(1.0 - ab_next - sigma ** 2) * eps_hat
            noise = torch.randn_like(x_t) if t_next > 0 else 0.0
            x_t = torch.sqrt(ab_next) * x0_pred + dir_xt + sigma * noise

    return x_t  # ≈ x_0
