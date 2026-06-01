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
    s: torch.Tensor,
    cs: torch.Tensor,
    shape: tuple | None = None,
    device: str = "cuda",
    S: int = 30,
    t_last: int = 4,
    r: float = 10.0,
    eta: float = 0.0,
    clamp_x0: bool = True,
) -> torch.Tensor:
    """
    DDIM-style deterministic sampling (eta=0).

    s, cs : [B, 1, H, W] conditioning (log-normalized).
    shape : 출력 텐서 shape. 기본은 s.shape.
    return: x_0 (log-normalized [0, 1] approx).
    """
    if shape is None:
        shape = tuple(s.shape)
    model.eval()
    schedule = schedule.to(device)

    x_t = torch.randn(shape, device=device)
    ts = dips_basic_schedule(T=schedule.T, S=S, t_last=t_last, r=r)

    for i in range(len(ts) - 1):
        t      = ts[i]
        t_next = ts[i + 1]
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_hat  = model(x_t, t_tensor, s, cs)

        ab_t    = schedule.alpha_bar[t]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        x0_pred = (x_t - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        if clamp_x0:
            x0_pred = x0_pred.clamp(0.0, 1.0)

        if eta == 0.0:
            dir_xt = torch.sqrt(1.0 - ab_next) * eps_hat
            x_t = torch.sqrt(ab_next) * x0_pred + dir_xt
        else:
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_next)
            dir_xt = torch.sqrt(1.0 - ab_next - sigma ** 2) * eps_hat
            noise = torch.randn_like(x_t) if t_next > 0 else 0.0
            x_t = torch.sqrt(ab_next) * x0_pred + dir_xt + sigma * noise

    return x_t  # ≈ x_0
