"""
renoise speckle 설계 sanity check. CPU 에서 실행 가능, 데이터/GPU 불필요.

    python scripts/sanity_check.py

정규화(r/μ/D_A) · 모델(in_ch=4 concat) · masked diffusion · 샘플러 · (있으면) dataset 검증.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nstsr.config.norm_config import (
    normalize_r, denormalize_r, normalize_mu_intensity, normalize_da, NORM_SPECKLE,
)
from nstsr.data.transforms import to_log10, from_log10
from nstsr.model.unet import UNet
from nstsr.diffusion.schedule import make_linear_schedule
from nstsr.diffusion.trainer import training_step
from nstsr.diffusion.sampler import sample as dips_sample

DEV = torch.device("cpu")
PASS, FAIL = "✅", "❌"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  {PASS if cond else FAIL} {name}" + (f"  — {detail}" if detail else ""))


print("=== 정규화 (r / μ / D_A) ===")
x = np.array([0.0, 1e-3, 1.0, 100.0], dtype=np.float32)
check("to_log10(0) 유한", np.isfinite(to_log10(x)).all(), f"log10(0)->{to_log10(x)[0]:.3f}")

L = NORM_SPECKLE["r_absmax"]
r = np.array([-L, -0.5, 0.0, 0.5, L], dtype=np.float32)
back = denormalize_r(normalize_r(r))
check("r norm↔denorm 왕복 |오차|<1e-5", np.abs(back - r).max() < 1e-5, f"max {np.abs(back-r).max():.2e}")
check("normalize_r(0)==0 (무변동)", abs(float(normalize_r(np.array([0.0], np.float32))[0])) < 1e-6)
check("normalize_r clip ∈ [-1,1]", float(normalize_r(np.array([10.0], np.float32))[0]) == 1.0)
mun = normalize_mu_intensity(np.array([1e-4, 1.0, 1e3], np.float32))
check("normalize_mu ∈ [0,1]", mun.min() >= 0 and mun.max() <= 1.0, f"[{mun.min():.2f},{mun.max():.2f}]")
dan = normalize_da(np.array([0.1, 1.0, 1000.0], np.float32))
check("normalize_da ∈ [0,1]", dan.min() >= 0 and dan.max() <= 1.0, f"[{dan.min():.2f},{dan.max():.2f}]")


print("\n=== 모델 (UNet in_ch=4, input-concat) ===")
B, P = 2, 64
xt = torch.randn(B, 1, P, P)
cond = torch.rand(B, 3, P, P)            # μ, D_A, shadow
t = torch.randint(0, 1000, (B,))
net = UNet(in_ch=4, out_ch=1, base_ch=32, ch_mults=(1, 2, 4, 8),
           num_res_blocks=2, attn_resolutions=(16,), use_mcam=False, input_resolution=P).to(DEV)
net.eval()
with torch.no_grad():
    out = net(xt, t, cond)
check("forward 출력 shape == (B,1,P,P)", tuple(out.shape) == (B, 1, P, P), f"{tuple(out.shape)}")
with torch.no_grad():
    o1 = net(xt, torch.zeros_like(t), cond)
    o2 = net(xt, torch.full_like(t, 999), cond)
check("t 바꾸면 출력 달라짐", not torch.allclose(o1, o2, atol=1e-5), f"mean|Δ|={(o1-o2).abs().mean():.2e}")
with torch.no_grad():
    c1 = net(xt, t, torch.zeros_like(cond))
    c2 = net(xt, t, torch.ones_like(cond))
check("cond 바꾸면 출력 달라짐", not torch.allclose(c1, c2, atol=1e-5), f"mean|Δ|={(c1-c2).abs().mean():.2e}")


print("\n=== diffusion (masked) / 학습 1-step ===")
sched = make_linear_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, device=DEV)
net.train()
cmask = (torch.rand(B, 1, P, P) > 0.1).float()         # ~90% valid
batch = {"r": torch.rand(B, 1, P, P) * 2 - 1, "cond": cond, "cmask": cmask}
loss = training_step(batch, net, sched, device=DEV)
loss.backward()
g = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
check("masked training_step loss 유한", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
check("backward gradient 흐름", float(g) > 0, f"Σ|grad|={float(g):.3e}")
# cmask=0 이면 loss 0 (마스킹 동작)
batch0 = {"r": batch["r"], "cond": cond, "cmask": torch.zeros_like(cmask)}
check("cmask=0 → loss 0 (마스킹)", float(training_step(batch0, net, sched, DEV)) == 0.0)

print("\n=== 샘플러 (cond → r̂) ===")
net.eval()
with torch.no_grad():
    rh = dips_sample(net, sched, cond, shape=(B, 1, P, P), device=DEV, S=6, t_last=1, eta=1.0)
check("sample 출력 shape", tuple(rh.shape) == (B, 1, P, P), f"{tuple(rh.shape)}")
check("sample 유한", torch.isfinite(rh).all().item())


print("\n=== dataset (있으면) ===")
DS = os.environ.get("SPECKLE_DS", "/media/sdb8TB/sentinel1/korea/speckle_ds")
if (Path(DS) / "splits" / "train.txt").exists():
    from nstsr.data.dataset import SARSpeckleDataset
    ds = SARSpeckleDataset(DS, "train", patch_size=128, augment=True)
    b = ds[0]
    check("dataset r/cond/cmask shape", b["r"].shape == (1, 128, 128) and b["cond"].shape == (3, 128, 128))
    check("dataset r ∈ [-1,1]", float(b["r"].min()) >= -1.001 and float(b["r"].max()) <= 1.001,
          f"[{float(b['r'].min()):.2f},{float(b['r'].max()):.2f}]")
else:
    print(f"  ⏭️  speckle_ds 없음 → 건너뜀 ({DS})")


print(f"\n{'='*50}")
n_pass = sum(results)
print(f"결과: {n_pass}/{len(results)} 통과")
sys.exit(0 if n_pass == len(results) else 1)
