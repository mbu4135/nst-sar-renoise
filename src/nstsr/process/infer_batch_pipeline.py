"""
InferBatchPipeline — 디렉토리 단위 일괄 추론 + overlap-tile blending.

CLI:
    python main.py infer-batch \
        --ckpt   checkpoints/vv_rnsd_baseline/ema.pt \
        --s_dir  /path/to/clean_dir \
        --out_dir /path/to/synth_dir \
        --pol vv --patch 256 --overlap 32
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
from nstsr.utils.io import load_image, save_image
from nstsr.utils.logger import get_logger


def _hann_window_2d(patch: int) -> np.ndarray:
    w1 = np.hanning(patch)
    return np.outer(w1, w1).astype(np.float32)


def _overlap_tile_infer(
    model: UNet,
    schedule,
    s_norm: np.ndarray,
    cs_norm: np.ndarray,
    patch: int,
    overlap: int,
    device: str,
    S: int,
    t_last: int,
    r: float,
    eta: float,
) -> np.ndarray:
    """log-normalized 입력 → log-normalized 출력. overlap-tile + Hanning blending.

    cs_norm 은 s_norm 보다 저해상(multilook)일 수 있어 타일별로 비율에 맞춰 자른다
    (모델 cs_enc 가 bottleneck 으로 resize 하므로 정확한 크기는 무관).
    """
    H, W = s_norm.shape
    ch, cw = cs_norm.shape
    cty = max(1, round(patch * ch / H))
    ctx = max(1, round(patch * cw / W))
    stride = patch - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than patch ({patch})")

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    win = _hann_window_2d(patch)

    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)

    for y0 in ys:
        for x0 in xs:
            s_tile  = s_norm [y0:y0+patch, x0:x0+patch]
            cy0 = min(int(round(y0 * ch / H)), ch - cty)
            cx0 = min(int(round(x0 * cw / W)), cw - ctx)
            cs_tile = cs_norm[cy0:cy0+cty, cx0:cx0+ctx]
            s_t  = torch.from_numpy(np.ascontiguousarray(s_tile)).float().unsqueeze(0).unsqueeze(0).to(device)
            cs_t = torch.from_numpy(np.ascontiguousarray(cs_tile)).float().unsqueeze(0).unsqueeze(0).to(device)
            x0_norm = dips_sample(
                model, schedule, s_t, cs_t, shape=s_t.shape, device=device,
                S=S, t_last=t_last, r=r, eta=eta,
            ).squeeze().detach().cpu().numpy()
            acc [y0:y0+patch, x0:x0+patch] += x0_norm * win
            wsum[y0:y0+patch, x0:x0+patch] += win

    wsum = np.where(wsum > 0, wsum, 1.0)
    return acc / wsum


class InferBatchPipeline:
    def __init__(self):
        self.logger = get_logger("nstsr.infer_batch")

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="batch inference (directory) with overlap-tile blending")
        p.add_argument("--ckpt", required=True)
        p.add_argument("--s_dir", required=True, help="clean (s) 영상 디렉토리")
        p.add_argument("--out_dir", required=True)
        p.add_argument("--cs_dir", default=None, help="(옵션) cs 영상 디렉토리 — 동일 stem 매칭")
        p.add_argument("--config", default=None)
        p.add_argument("--pol", default="vv")
        p.add_argument("--patch", type=int, default=256)
        p.add_argument("--overlap", type=int, default=32)
        p.add_argument("--steps", type=int, default=30)
        p.add_argument("--t_last", type=int, default=4)
        p.add_argument("--r", type=float, default=10.0)
        p.add_argument("--eta", type=float, default=0.0)
        p.add_argument("--ext", default=".tif")
        p.add_argument("--gpu", type=int, default=0)
        return p

    def _load_model(self, cfg, ckpt_path: str, device: str) -> UNet:
        m = cfg.model
        model = UNet(
            in_ch=m.in_ch, out_ch=m.out_ch, base_ch=m.base_ch,
            ch_mults=tuple(m.ch_mults), num_res_blocks=m.num_res_blocks,
            attn_resolutions=tuple(m.attn_resolutions), use_mcam=m.use_mcam,
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(state)
        model.eval()
        return model

    def run(self, args: argparse.Namespace | None = None) -> None:
        if args is None:
            args = self.build_parser().parse_args()

        cfg = load_config(args.config)
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        s_dir   = Path(args.s_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cs_dir  = Path(args.cs_dir) if args.cs_dir else None

        model = self._load_model(cfg, args.ckpt, device)
        schedule = make_linear_schedule(
            T=cfg.diffusion.T, beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end, device=device,
        )

        s_paths = [p for p in sorted(s_dir.iterdir()) if p.suffix.lower() == args.ext.lower()]
        self.logger.info(f"{len(s_paths)} files in {s_dir}")

        for sp in s_paths:
            s_lin = load_image(sp).astype(np.float32)
            s_log = to_log10(s_lin)
            s_norm = normalize_image(s_log, pol=args.pol)

            if cs_dir is not None:
                cp = cs_dir / sp.name
                if not cp.exists():
                    self.logger.warning(f"no cs match for {sp.name} — skipping")
                    continue
                cs_norm = load_image(cp).astype(np.float32)   # 저해상/ full-res 모두 허용
            else:
                # 무변화: 저해상 상수 0.5 (full-res cs_enc conv 낭비 회피)
                H0, W0 = s_norm.shape
                cs_norm = np.full((max(1, H0 // 16), max(1, W0 // 16)), 0.5, dtype=np.float32)

            x0_norm = _overlap_tile_infer(
                model, schedule, s_norm, cs_norm,
                patch=args.patch, overlap=args.overlap, device=device,
                S=args.steps, t_last=args.t_last, r=args.r, eta=args.eta,
            )
            x0_norm = np.clip(x0_norm, 0.0, 1.0)
            y_log = denormalize_image(x0_norm, pol=args.pol)
            y_lin = np.clip(from_log10(y_log), 0.0, None)

            out_path = out_dir / sp.name
            save_image(out_path, y_lin)
            self.logger.info(f"  {sp.name} → {out_path.name}  shape={y_lin.shape}")
