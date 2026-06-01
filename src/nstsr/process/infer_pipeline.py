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

from tqdm import tqdm

from nstsr.config.config import load_config
from nstsr.config.norm_config import denormalize_image, normalize_image, normalize_ratio
from nstsr.data.ratio_builder import _patch_offsets, build_ratio_cs_patched_numpy
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
        p.add_argument("--cs", default=None,
                       help="(옵션) **이미 정규화된** [0,1] cs 영상 (0.5=무변화). 생략 시 0.5 상수")
        p.add_argument("--cs_raw", default=None,
                       help="(옵션) **linear ratio** 영상 → 내부에서 normalize_ratio 로 cs 생성. --cs 와 배타")
        p.add_argument("--ratio_ckpt", default=None,
                       help="(옵션) --cs_raw 와 함께 주면 i2i_unetsar ratio 모델로 denoise 까지 (학습 cs 와 동일 경로)")
        p.add_argument("--ratio_base_ch", type=int, default=32, help="ratio 모델 UNet 폭 (i2i_ratio_C11_ft=32)")
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
        # ── overlap-tile (큰 영상을 학습 패치 크기로 잘라 추론 후 합침) ──
        p.add_argument("--tile", type=int, default=None,
                       help="타일 크기. 기본=cfg.data.patch_size(학습 패치). 영상이 더 크면 자동 타일링")
        p.add_argument("--tile_stride", type=int, default=None, help="타일 stride (기본 tile//2)")
        p.add_argument("--tile_batch", type=int, default=16, help="한 번에 추론할 타일 수")
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

    # ── 샘플링: 작은 영상은 통째로, 큰 영상은 타일링 후 Hanning 합성 ──────
    @torch.no_grad()
    def _sample_whole(self, model, schedule, s_norm, cs_norm, device, args):
        s_t  = torch.from_numpy(np.ascontiguousarray(s_norm)).float()[None, None].to(device)
        cs_t = torch.from_numpy(np.ascontiguousarray(cs_norm)).float()[None, None].to(device)
        x0 = dips_sample(model, schedule, s_t, cs_t, shape=s_t.shape, device=device,
                         S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta)
        return x0.squeeze().detach().cpu().numpy()

    @torch.no_grad()
    def _sample_tiled(self, model, schedule, s_norm, cs_norm, device, tile, stride, batch, args):
        """학습 패치 크기(tile)로 잘라 타일별 추론 → Hanning overlap-tile 합성 (norm 도메인)."""
        H, W = s_norm.shape
        ys = _patch_offsets(H, tile, stride)
        xs = _patch_offsets(W, tile, stride)
        coords = [(y0, x0) for y0 in ys for x0 in xs]
        hann = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float64)
        accum = np.zeros((H, W), dtype=np.float64)
        wsum  = np.zeros((H, W), dtype=np.float64)
        for i in tqdm(range(0, len(coords), batch), desc="tiles", unit="batch"):
            chunk = coords[i:i + batch]
            sb = np.stack([s_norm[y:y + tile, x:x + tile] for (y, x) in chunk])
            cb = np.stack([cs_norm[y:y + tile, x:x + tile] for (y, x) in chunk])
            s_t  = torch.from_numpy(sb).float().unsqueeze(1).to(device)
            cs_t = torch.from_numpy(cb).float().unsqueeze(1).to(device)
            x0 = dips_sample(model, schedule, s_t, cs_t, shape=s_t.shape, device=device,
                             S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta)
            x0 = x0[:, 0].detach().cpu().numpy()
            for j, (y, x) in enumerate(chunk):
                accum[y:y + tile, x:x + tile] += x0[j] * hann
                wsum[y:y + tile, x:x + tile]  += hann
        wsum = np.where(wsum == 0, 1.0, wsum)
        return (accum / wsum).astype(np.float32)

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

        # ── load s, build cs (norm 도메인 [H, W] numpy) ───────────────
        shape = tuple(args.img_shape) if args.img_shape else None
        s_lin = self._load_img(args.s, shape)
        s_norm = normalize_image(to_log10(s_lin), pol=args.pol).astype(np.float32)
        H, W = s_norm.shape
        if args.cs is not None and args.cs_raw is not None:
            raise ValueError("--cs 와 --cs_raw 는 함께 쓸 수 없습니다 (전자는 정규화 완료, 후자는 linear)")

        if args.cs is not None:
            # 이미 정규화된 [0,1] cs 그대로 사용
            cs_norm = self._load_img(args.cs, shape).astype(np.float32)
        elif args.cs_raw is not None:
            # linear ratio → cs 변환
            ratio_lin = self._load_img(args.cs_raw, shape).astype(np.float32)
            if ratio_lin.shape != s_norm.shape:
                raise ValueError(f"s shape {s_norm.shape} != cs_raw shape {ratio_lin.shape}")
            if args.ratio_ckpt:
                # normalize + i2i ratio 모델 denoise (학습 cs 와 동일 경로). s=1 로 두면 ratio 그대로 사용됨.
                from nstsr.model.ratio_denoiser import build_ratio_denoiser
                rd = build_ratio_denoiser(arch="i2i_unetsar", ckpt_path=args.ratio_ckpt,
                                          device=device, base_ch=args.ratio_base_ch)
                cs_norm = build_ratio_cs_patched_numpy(
                    ratio_lin, np.ones_like(ratio_lin), ratio_denoiser=rd,
                    patch_size=512, stride=256, pol=args.pol, device=device,
                )
                self.logger.info("cs_raw → normalize_ratio + i2i denoise")
            else:
                m = ratio_lin > 0
                cs_norm = normalize_ratio(np.log10(np.where(m, ratio_lin, 1.0)), pol=args.pol).astype(np.float32)
                cs_norm[~m] = 0.0
                self.logger.info("cs_raw → normalize_ratio (denoise 없음)")
        else:
            cs_norm = np.full((H, W), 0.5, dtype=np.float32)
            self.logger.info("cs not provided — using constant 0.5 (no-change scenario)")

        # ── model (input_resolution = 타일 크기 = 학습 패치) ───────────
        tile = int(args.tile) if args.tile else int(cfg.data.patch_size)
        tile = min(tile, H, W)
        model = self._load_model(cfg, args.ckpt, device, input_resolution=tile)
        schedule = make_linear_schedule(
            T=cfg.diffusion.T, beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end, device=device,
        )

        # ── sample: 영상이 타일보다 크면 overlap-tile, 아니면 통째로 ──
        if H <= tile and W <= tile:
            self.logger.info(f"whole-image sampling: shape=({H},{W}) steps={args.steps}")
            x0_norm = self._sample_whole(model, schedule, s_norm, cs_norm, device, args)
        else:
            stride = int(args.tile_stride) if args.tile_stride else max(1, tile // 2)
            self.logger.info(f"tiled sampling: ({H},{W}) tile={tile} stride={stride} "
                             f"batch={args.tile_batch} steps={args.steps}")
            x0_norm = self._sample_tiled(model, schedule, s_norm, cs_norm, device,
                                         tile, stride, args.tile_batch, args)

        # ── log-norm → linear ─────────────────────────────────────────
        y_lin = np.clip(from_log10(denormalize_image(x0_norm, pol=args.pol)), 0.0, None)
        self._save_img(args.out, y_lin)
        self.logger.info(f"saved: {args.out}  shape={y_lin.shape}  mean={y_lin.mean():.4g}  std={y_lin.std():.4g}")
