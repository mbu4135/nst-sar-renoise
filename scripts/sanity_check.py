"""
SAR-RNSD 구현 sanity check (스펙 §6). CPU 에서 실행 가능.

    python scripts/sanity_check.py

데이터/ GPU 없이 모델·정규화·diffusion·ratio denoiser 연결을 검증한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nstsr.config.norm_config import (
    normalize_image, denormalize_image, normalize_ratio,
)
from nstsr.data.transforms import to_log10, from_log10
from nstsr.data.ratio_builder import build_ratio_cs
from nstsr.model.unet import UNet
from nstsr.model.ratio_denoiser import build_ratio_denoiser
from nstsr.diffusion.schedule import make_linear_schedule
from nstsr.diffusion.trainer import training_step

# filtering I2I ratio 체크포인트. 다른 머신에선 환경변수로 지정:
#   RATIO_CKPT=/path/to/best_model.pth python scripts/sanity_check.py
# 없으면 ratio-denoiser 섹션만 건너뛰고 나머지 검사는 그대로 수행.
import os
RATIO_CKPT = os.environ.get(
    "RATIO_CKPT",
    "/media/sdb8TB/naraspace/nst-sar-filtering/checkpoints/i2i_ratio_C11_ft/best_model.pth",
)

DEV = torch.device("cpu")
PASS, FAIL = "✅", "❌"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  {PASS if cond else FAIL} {name}" + (f"  — {detail}" if detail else ""))


print("=== §6.1 데이터 파이프라인 ===")
x = np.array([0.0, 1e-3, 1.0, 100.0], dtype=np.float32)
lg = to_log10(x)
check("to_log10(0) NaN/inf 없음", np.isfinite(lg).all(), f"log10(0)->{lg[0]:.3f}")
rt = from_log10(to_log10(x))
check("log10 왕복 오차<1e-3", np.allclose(rt[1:], x[1:], rtol=1e-3), f"max err {np.abs(rt[1:]-x[1:]).max():.2e}")

xl = np.linspace(-12, 10, 50).astype(np.float32)
back = denormalize_image(normalize_image(xl, pol="vv"), pol="vv")
check("image norm↔denorm 왕복 |오차|<1e-5", np.abs(back - xl).max() < 1e-5, f"max {np.abs(back-xl).max():.2e}")

nr = normalize_ratio(np.array([0.0], dtype=np.float32), pol="vv")  # log10(1.0)=0
check("normalize_ratio(log10 1.0)==0.5", abs(float(nr[0]) - 0.5) < 1e-6, f"={float(nr[0]):.6f}")


print("\n=== §6.2 모델 ===")
B, P = 2, 64
xt = torch.randn(B, 1, P, P)
s  = torch.rand(B, 1, P, P)
cs = torch.rand(B, 1, P, P)
t  = torch.randint(0, 1000, (B,))
net = UNet(in_ch=1, out_ch=1, base_ch=32, ch_mults=(1, 2, 4, 8),
           num_res_blocks=2, attn_resolutions=(16,), use_mcam=True, input_resolution=P).to(DEV)
net.eval()
with torch.no_grad():
    out = net(xt, t, s, cs)
check("UNet forward 출력 shape == 입력", tuple(out.shape) == (B, 1, P, P), f"{tuple(out.shape)}")
check("MCAM enc_s, enc_cs 가중치 비공유", id(net.mcam.enc_s) != id(net.mcam.enc_cs))
with torch.no_grad():
    o_t1 = net(xt, torch.zeros_like(t), s, cs)
    o_t2 = net(xt, torch.full_like(t, 999), s, cs)
check("t 바꾸면 출력 달라짐", not torch.allclose(o_t1, o_t2, atol=1e-5),
      f"mean|Δ|={ (o_t1-o_t2).abs().mean():.2e}")
with torch.no_grad():
    o_cs1 = net(xt, t, s, torch.zeros_like(cs))
    o_cs2 = net(xt, t, s, torch.ones_like(cs))
check("cs 바꾸면 출력 달라짐 (조건부 검증)", not torch.allclose(o_cs1, o_cs2, atol=1e-5),
      f"mean|Δ|={ (o_cs1-o_cs2).abs().mean():.2e}")


print("\n=== §6.3 diffusion / 학습 1-step ===")
sched = make_linear_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, device=DEV)
net.train()
batch = {"y": torch.rand(B, 1, P, P), "s": s, "cs": cs}
loss = training_step(batch, net, sched, device=DEV)
loss.backward()
g = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
check("training_step loss 유한", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
check("backward 후 gradient 흐름", float(g) > 0, f"Σ|grad|={float(g):.3e}")


print("\n=== ratio denoiser 연결 (i2i_unetsar, 실제 체크포인트) ===")
if not Path(RATIO_CKPT).exists():
    print(f"  ⏭️  ratio ckpt 없음 → 이 섹션 건너뜀 (훈련엔 불필요; prepare 때만 필요)")
    print(f"      필요시: RATIO_CKPT=/path/best_model.pth 로 지정  (현재: {RATIO_CKPT})")
else:
    rd = build_ratio_denoiser(arch="i2i_unetsar", ckpt_path=RATIO_CKPT, device="cpu", base_ch=32)
    check("ratio denoiser eval 모드", not rd.training)
    check("ratio denoiser 전부 freeze", all(not p.requires_grad for p in rd.parameters()))
    # cs 빌더 end-to-end: 작은 linear y,s -> cs
    y_lin = torch.rand(1, 1, 256, 256) * 5 + 0.1
    s_lin = torch.rand(1, 1, 256, 256) * 5 + 0.1
    with torch.no_grad():
        cs_out = build_ratio_cs(y_lin, s_lin, ratio_denoiser=rd, pol="vv")
    check("build_ratio_cs 출력 shape", tuple(cs_out.shape) == (1, 1, 256, 256), f"{tuple(cs_out.shape)}")
    check("cs 출력 유한 & 대략 [0,1]", torch.isfinite(cs_out).all().item() and
          float(cs_out.min()) > -0.5 and float(cs_out.max()) < 1.5,
          f"[{float(cs_out.min()):.3f}, {float(cs_out.max()):.3f}]")


print(f"\n{'='*50}")
n_pass = sum(results)
print(f"결과: {n_pass}/{len(results)} 통과")
sys.exit(0 if n_pass == len(results) else 1)
