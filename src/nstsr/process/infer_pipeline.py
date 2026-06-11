"""
InferPipeline — renoise speckle 합성.

조건 (대상 장면의 시간통계 + 기하; 모두 같은 격자):
    --mu      temporal mean intensity (μ, linear)         [필수]
    --da      amplitude dispersion D_A = 1/MSR             [옵션, 없으면 1.0 상수]
    --shadow  valid mask (1=정상, 0=shadow/no-data)        [옵션, 없으면 all-valid]

모델은 r = log10(y/μ) (speckle) 를 cond=[μ_n, D_A_n, valid] 조건으로 생성하고,
출력은 ŷ = μ · 10**r̂ (single-look noisy intensity). shadow/no-data 픽셀은 0.

큰 영상은 학습 패치 크기로 overlap-tile 후 Hanning 합성 (r 도메인). eta>0 권장(speckle 텍스처).

I/O: .img(big-endian float32, --img_shape 필요) / .npy / .tif.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nstsr.config.config import load_config
from nstsr.config.norm_config import (
    normalize_mu_intensity, normalize_da, denormalize_r,
)
from nstsr.data.ratio_builder import _patch_offsets
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
        p = argparse.ArgumentParser(description="renoise speckle 합성 (μ,D_A,shadow → noisy ŷ)")
        p.add_argument("--ckpt", required=True, help="ema.pt or last.pt")
        p.add_argument("--mu", required=True, help="temporal mean intensity μ (linear .npy/.tif/.img)")
        p.add_argument("--da", default=None, help="(옵션) D_A=1/MSR map. 생략 시 1.0 상수")
        p.add_argument("--shadow", default=None, help="(옵션) valid mask(1=정상). 생략 시 all-valid")
        p.add_argument("--out", required=True, help="출력 noisy intensity ŷ (.npy/.tif/.img)")
        p.add_argument("--img_shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                       help=".img I/O 시 영상 크기 (예: --img_shape 13124 68647)")
        p.add_argument("--config", default=None)
        p.add_argument("--steps", type=int, default=30)
        p.add_argument("--t_last", type=int, default=1)
        p.add_argument("--r", type=float, default=10.0)
        p.add_argument("--eta", type=float, default=1.0, help="stochastic 샘플링(speckle 텍스처). 0=deterministic")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--tile", type=int, default=None, help="타일 크기. 기본=cfg.data.patch_size")
        p.add_argument("--tile_stride", type=int, default=None, help="타일 stride (기본 tile//2)")
        p.add_argument("--tile_batch", type=int, default=16)
        return p

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

    # ── 샘플링: cond(3ch) → r̂(log10 ratio). 큰 영상은 overlap-tile ─────────
    @torch.no_grad()
    def _sample_r(self, model, schedule, cond, device, tile, stride, batch, args):
        """cond: [3, H, W] numpy → r̂ [H, W] (log10 ratio, denormalized)."""
        _, H, W = cond.shape
        if H <= tile and W <= tile:
            ct = torch.from_numpy(np.ascontiguousarray(cond)).float()[None].to(device)
            r0 = dips_sample(model, schedule, ct, shape=(1, 1, H, W), device=device,
                             S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta)
            return denormalize_r(r0[0, 0].cpu().numpy())

        ys = _patch_offsets(H, tile, stride)
        xs = _patch_offsets(W, tile, stride)
        coords = [(y0, x0) for y0 in ys for x0 in xs]
        hann = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float64)
        accum = np.zeros((H, W), np.float64)
        wsum = np.zeros((H, W), np.float64)
        for i in tqdm(range(0, len(coords), batch), desc="tiles", unit="batch"):
            chunk = coords[i:i + batch]
            cb = np.stack([cond[:, y:y + tile, x:x + tile] for (y, x) in chunk])  # [b,3,tile,tile]
            ct = torch.from_numpy(cb).float().to(device)
            r0 = dips_sample(model, schedule, ct, shape=(len(chunk), 1, tile, tile),
                             device=device, S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta)
            r0 = denormalize_r(r0[:, 0].cpu().numpy())
            for j, (y, x) in enumerate(chunk):
                accum[y:y + tile, x:x + tile] += r0[j] * hann
                wsum[y:y + tile, x:x + tile] += hann
        wsum = np.where(wsum == 0, 1.0, wsum)
        return (accum / wsum).astype(np.float32)

    def _load_model(self, cfg, ckpt_path, device, input_resolution):
        m = cfg.model
        model = UNet(in_ch=m.in_ch, out_ch=m.out_ch, base_ch=m.base_ch,
                     ch_mults=tuple(m.ch_mults), num_res_blocks=m.num_res_blocks,
                     attn_resolutions=tuple(m.attn_resolutions), use_mcam=m.use_mcam,
                     input_resolution=input_resolution).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
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
        shape = tuple(args.img_shape) if args.img_shape else None

        # ── conditioning 영상 로드 → cond [3, H, W] ───────────────────
        mu = self._load_img(args.mu, shape)
        H, W = mu.shape
        valid = ((self._load_img(args.shadow, shape) > 0.5) if args.shadow else np.ones((H, W))).astype(np.float32)
        da = self._load_img(args.da, shape) if args.da else np.ones((H, W), np.float32)
        mu_n = normalize_mu_intensity(mu).astype(np.float32)
        da_n = normalize_da(da).astype(np.float32)
        cond = np.stack([mu_n, da_n, valid]).astype(np.float32)   # [3, H, W]
        self.logger.info(f"cond: μ{mu.shape} D_A{'(const1)' if not args.da else ''} "
                         f"valid{'(all)' if not args.shadow else ''}  shape={cond.shape}")

        tile = int(args.tile) if args.tile else int(cfg.data.patch_size)
        tile = min(tile, H, W)
        stride = int(args.tile_stride) if args.tile_stride else max(1, tile // 2)
        model = self._load_model(cfg, args.ckpt, device, input_resolution=tile)
        schedule = make_linear_schedule(T=cfg.diffusion.T, beta_start=cfg.diffusion.beta_start,
                                        beta_end=cfg.diffusion.beta_end, device=device)

        self.logger.info(f"sampling: ({H},{W}) tile={tile} stride={stride} "
                         f"steps={args.steps} eta={args.eta}")
        r_hat = self._sample_r(model, schedule, cond, device, tile, stride, args.tile_batch, args)

        # ── 복원: ŷ = μ · 10**r̂, shadow/no-data → 0 ──────────────────
        y_lin = (mu * np.power(10.0, r_hat)).astype(np.float32)
        y_lin[valid < 0.5] = 0.0
        y_lin[mu <= 0] = 0.0
        self._save_img(args.out, y_lin)
        self.logger.info(f"saved: {args.out}  shape={y_lin.shape}  "
                         f"mean={y_lin[y_lin>0].mean():.4g}  std={y_lin[y_lin>0].std():.4g}")
