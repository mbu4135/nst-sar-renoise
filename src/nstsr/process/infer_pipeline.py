"""
InferPipeline — 단일 clean s 영상 → synthetic single-look noisy 영상.

CLI:
    python main.py infer \
        --ckpt checkpoints/vv_rnsd_baseline/ema.pt \
        --s    sample_clean_vv.tif \
        --out  sample_synth_noisy_vv.tif \
        --pol  vv --steps 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nstsr.config.config import load_config
from nstsr.config.norm_config import denormalize_image, normalize_image
from nstsr.data.transforms import from_log10, to_log10
from nstsr.diffusion.sampler import sample as dips_sample
from nstsr.diffusion.schedule import make_linear_schedule
from nstsr.model.unet import UNet
from nstsr.utils.io import load_image, save_image, load_raw_bef32, save_raw_bef32
from nstsr.utils.logger import get_logger


class InferPipeline:
    def __init__(self):
        self.logger = get_logger("nstsr.infer")

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="infer single image")
        p.add_argument("--ckpt", required=True, help="ema.pt or last.pt")
        p.add_argument("--s", required=True, help="clean (s) 영상 경로 (linear .npy/.tif/.img)")
        p.add_argument("--cs", default=None, help="(옵션) ratio cs 영상. 생략 시 0.5 상수")
        p.add_argument("--out", required=True, help="출력 노이지 영상 경로 (.npy/.tif/.img)")
        p.add_argument("--img_shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                       help=".img(raw big-endian float32) 입출력 시 영상 크기 (예: --img_shape 3680 12960)")
        p.add_argument("--config", default=None)
        p.add_argument("--pol", default="vv")
        p.add_argument("--steps", type=int, default=30)
        p.add_argument("--t_last", type=int, default=4)
        p.add_argument("--r", type=float, default=10.0)
        p.add_argument("--eta", type=float, default=0.0)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--seed", type=int, default=None)
        return p

    # ── .npy/.tif/.img 공용 I/O (.img 는 --img_shape 필요) ──────────────
    @staticmethod
    def _load_img(path, shape):
        if str(path).lower().endswith(".img"):
            if shape is None:
                raise ValueError(f"{path}: .img 입력엔 --img_shape H W 가 필요합니다")
            return load_raw_bef32(path, shape).astype(np.float32)
        return load_image(path).astype(np.float32)

    @staticmethod
    def _save_img(path, arr):
        if str(path).lower().endswith(".img"):
            save_raw_bef32(path, arr)
        else:
            save_image(path, arr)

    # ─────────────────────────────────────────────────────────────────
    def _load_model(self, cfg, ckpt_path: str, device: str, input_resolution: int) -> UNet:
        m = cfg.model
        model = UNet(
            in_ch=m.in_ch, out_ch=m.out_ch, base_ch=m.base_ch,
            ch_mults=tuple(m.ch_mults), num_res_blocks=m.num_res_blocks,
            attn_resolutions=tuple(m.attn_resolutions), use_mcam=m.use_mcam,
            input_resolution=input_resolution,
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(state)
        model.eval()
        return model

    # ─────────────────────────────────────────────────────────────────
    def run(self, args: argparse.Namespace | None = None) -> None:
        if args is None:
            args = self.build_parser().parse_args()
        if args.seed is not None:
            torch.manual_seed(args.seed)

        cfg = load_config(args.config)
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

        # ── load s, build cs ──────────────────────────────────────────
        shape = tuple(args.img_shape) if args.img_shape else None
        s_lin = self._load_img(args.s, shape)
        s_log = to_log10(s_lin)
        s_norm = normalize_image(s_log, pol=args.pol)
        s_t = torch.from_numpy(np.ascontiguousarray(s_norm)).float().unsqueeze(0).unsqueeze(0).to(device)

        if args.cs is None:
            cs_t = torch.full_like(s_t, 0.5)
            self.logger.info("cs not provided — using constant 0.5 (no-change scenario)")
        else:
            cs_arr = self._load_img(args.cs, shape)
            cs_t = torch.from_numpy(np.ascontiguousarray(cs_arr)).float().unsqueeze(0).unsqueeze(0).to(device)

        if s_t.shape != cs_t.shape:
            raise ValueError(f"s shape {s_t.shape} != cs shape {cs_t.shape}")

        # ── model ─────────────────────────────────────────────────────
        H, W = s_t.shape[-2:]
        model = self._load_model(cfg, args.ckpt, device, input_resolution=min(H, W))
        schedule = make_linear_schedule(
            T=cfg.diffusion.T, beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end, device=device,
        )

        # ── sample ────────────────────────────────────────────────────
        self.logger.info(f"sampling: shape={tuple(s_t.shape)} steps={args.steps}")
        x0_norm = dips_sample(
            model, schedule, s_t, cs_t, shape=s_t.shape, device=device,
            S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta,
        )

        # ── log-norm → linear ─────────────────────────────────────────
        x0_norm_np = x0_norm.squeeze().detach().cpu().numpy()
        y_log = denormalize_image(x0_norm_np, pol=args.pol)
        y_lin = from_log10(y_log)
        y_lin = np.clip(y_lin, 0.0, None)  # 음수 방지

        self._save_img(args.out, y_lin)
        self.logger.info(f"saved: {args.out}  shape={y_lin.shape}  mean={y_lin.mean():.4g}  std={y_lin.std():.4g}")
