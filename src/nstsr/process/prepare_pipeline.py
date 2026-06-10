"""
PreparePipeline — raw (y, s) → (y, s, cs) triple 을 디스크에 저장하고 split 파일 생성.

cs 는 기본 multilook 모드: linear ratio = y/s 를 looks×looks 블록평균(고전 multilook)
→ log10 → symmetric min-max(`norm_config.ratio_ml`, vv L=0.6) 로 만들어 patch/looks
해상도(예 512/16=32)로 저장. 학습 denoiser 불필요. (--cs_mode denoiser 로 구 경로 선택)

CLI:
    python main.py prepare \
        --y_dir /USER/single_look_vv \
        --s_dir /USER/temporal_avg_vv \
        --out_dir ./data_root \
        --pol vv --channel C11 --img_shape 3680 12960 \
        --patch 512 --patch_stride 256 --cs_mode multilook --looks 16
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from nstsr.data.ratio_builder import (
    build_ratio_cs_numpy, build_ratio_cs_patched_numpy,
    build_ratio_cs_multilook_numpy, _patch_offsets,
)
from nstsr.model.ratio_denoiser import build_ratio_denoiser
from nstsr.utils.io import load_image, load_raw_bef32, save_npy
from nstsr.utils.logger import get_logger


class PreparePipeline:
    """raw 영상 디렉토리 → data_root/scenes/<id>/{y,s,cs}.npy + splits/{train,val}.txt."""

    def __init__(self):
        self.logger = get_logger("nstsr.prepare")

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="prepare (y, s, cs) triple dataset")
        p.add_argument("--y_dir", required=True, help="single-look (y) 영상 디렉토리")
        p.add_argument("--s_dir", required=True, help="temporal-averaged (s) 영상 디렉토리")
        p.add_argument("--out_dir", required=True, help="data_root (출력 디렉토리)")
        p.add_argument("--cs_mode", default="multilook", choices=["multilook", "denoiser"],
                       help="cs 생성 방식 (기본 multilook): multilook(고전 looks×looks 평균, 모델 불필요, cs=patch/looks 해상도) | denoiser(학습된 ratio 모델, cs=patch 해상도)")
        p.add_argument("--looks", type=int, default=16, help="multilook 모드 블록 크기 (cs 해상도 = patch/looks)")
        p.add_argument("--ratio_arch", default="identity", help="ratio denoiser arch (예: i2i_unetsar)")
        p.add_argument("--ratio_ckpt", default=None, help="pretrained ratio denoiser 체크포인트")
        p.add_argument("--ratio_base_ch", type=int, default=32, help="i2i_unetsar UNet 폭")
        p.add_argument("--pol", default="vv", choices=["vv", "vh", "hh", "hv"])
        p.add_argument("--eps", type=float, default=1e-8)
        p.add_argument("--device", default="cuda")
        p.add_argument("--val_ratio", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--ext", default=".npy", help="(flat 모드) 입력 확장자 .npy/.tif")
        # ── raw .img 모드 (y=<y_dir>/<date>/<channel>.img, s=<s_dir>/<channel>.img 단일 공유) ──
        p.add_argument("--img_shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                       help="설정 시 raw big-endian float32 .img 모드 (예: --img_shape 3680 12960)")
        p.add_argument("--channel", default="C11", help="raw .img 모드에서 읽을 채널 파일명 (C11/C22/C12_mag)")
        p.add_argument("--ratio_patch", type=int, default=512, help="cs patch-based 추론 타일 크기")
        p.add_argument("--ratio_stride", type=int, default=256, help="cs patch-based 추론 stride")
        # ── 날짜 부분 선택 (raw .img 모드) ──
        p.add_argument("--dates", nargs="*", default=None,
                       help="처리할 날짜만 지정 (예: --dates 20180104 20180116). 생략 시 전부")
        p.add_argument("--limit", type=int, default=None,
                       help="앞에서 N개 날짜만 처리 (--dates 와 함께 쓰면 그 목록에서 N개)")
        # ── patch 단위 저장 + land 필터 (raw .img 모드) ──
        p.add_argument("--patch", type=int, default=None,
                       help="설정 시 scene 을 patch 로 잘라 저장 (예: 512). 미설정 시 full scene per date")
        p.add_argument("--patch_stride", type=int, default=None, help="patch stride (기본 = patch)")
        p.add_argument("--land_ref_date", default=None,
                       help="land mask 기준 날짜 (기본 = 처리 대상 첫 날짜). 그 날의 channel>0 가 land")
        p.add_argument("--land_min", type=float, default=0.70,
                       help="patch 채택 최소 land 비율 (기본 0.70)")
        return p

    # ─────────────────────────────────────────────────────────────────
    def _img_pairs(self, y_dir: Path, s_dir: Path, channel: str,
                   dates=None, limit=None):
        """raw .img 모드: (date, y_path, s_path) — s 는 단일 공유 평균.

        dates : 처리할 날짜 화이트리스트 (None 이면 전부).
        limit : 정렬 후 앞에서 N개만.
        """
        s_path = s_dir / f"{channel}.img"
        if not s_path.exists():
            raise FileNotFoundError(f"shared s image not found: {s_path}")

        def _is_date(name: str) -> bool:  # YYYYMMDD 폴더만 (ave_dir/ratio 등 제외)
            return len(name) == 8 and name.isdigit()

        wanted = set(dates) if dates else None
        pairs = []
        for sub in sorted(p for p in y_dir.iterdir() if p.is_dir() and _is_date(p.name)):
            if wanted is not None and sub.name not in wanted:
                continue
            yp = sub / f"{channel}.img"
            if yp.exists():
                pairs.append((sub.name, yp, s_path))
        if wanted:
            missing = sorted(wanted - {p[0] for p in pairs})
            if missing:
                self.logger.warning(f"requested dates with no {channel}.img: {missing}")
        if limit is not None:
            pairs = pairs[:limit]
        if not pairs:
            raise RuntimeError(f"no matching <YYYYMMDD>/{channel}.img found under {y_dir}")
        return pairs

    # ─────────────────────────────────────────────────────────────────
    def _pair_files(self, y_dir: Path, s_dir: Path, ext: str) -> List[Tuple[str, Path, Path]]:
        """
        파일명 stem 매칭으로 (scene_id, y_path, s_path) 목록 생성.
        y_dir / s_dir 의 파일명 stem 은 동일해야 한다.
        """
        y_files = {p.stem: p for p in sorted(y_dir.iterdir()) if p.suffix.lower() == ext.lower()}
        s_files = {p.stem: p for p in sorted(s_dir.iterdir()) if p.suffix.lower() == ext.lower()}
        common = sorted(set(y_files) & set(s_files))
        missing_in_s = sorted(set(y_files) - set(s_files))
        missing_in_y = sorted(set(s_files) - set(y_files))
        if missing_in_s:
            self.logger.warning(f"{len(missing_in_s)} files in y_dir without s match (first: {missing_in_s[:3]})")
        if missing_in_y:
            self.logger.warning(f"{len(missing_in_y)} files in s_dir without y match (first: {missing_in_y[:3]})")
        return [(k, y_files[k], s_files[k]) for k in common]

    # ─────────────────────────────────────────────────────────────────
    def _land_patch_coords(self, land_mask, patch: int, stride: int, land_min: float):
        """land_mask(bool [H,W]) 에서 land 비율 ≥ land_min 인 patch 좌상단 (y0,x0) 목록."""
        H, W = land_mask.shape
        ys = _patch_offsets(H, patch, stride)
        xs = _patch_offsets(W, patch, stride)
        coords = []
        for y0 in ys:
            for x0 in xs:
                if land_mask[y0:y0 + patch, x0:x0 + patch].mean() >= land_min:
                    coords.append((y0, x0))
        return coords

    # ─────────────────────────────────────────────────────────────────
    def run(self, args: argparse.Namespace | None = None) -> None:
        if args is None:
            args = self.build_parser().parse_args()

        y_dir = Path(args.y_dir)
        s_dir = Path(args.s_dir)
        out_dir = Path(args.out_dir)
        scenes_dir = out_dir / "scenes"
        splits_dir = out_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        multilook = args.cs_mode == "multilook"
        if multilook:
            ratio_model = None
            self.logger.info(f"cs_mode=multilook (looks={args.looks}) — denoiser 미사용, "
                             f"cs 해상도 = patch/{args.looks}")
        else:
            ratio_model = build_ratio_denoiser(
                arch=args.ratio_arch, ckpt_path=args.ratio_ckpt,
                device=args.device, base_ch=args.ratio_base_ch,
            )

        img_mode = args.img_shape is not None
        if img_mode:
            shape = (int(args.img_shape[0]), int(args.img_shape[1]))
            pairs = self._img_pairs(y_dir, s_dir, args.channel, dates=args.dates, limit=args.limit)
            self.logger.info(f"raw .img mode: {len(pairs)} dates, channel={args.channel}, "
                             f"shape={shape}, shared s={pairs[0][2]}")
            s_cache = load_raw_bef32(pairs[0][2], shape)   # 단일 공유 평균 — 한 번만 로드
        else:
            pairs = self._pair_files(y_dir, s_dir, args.ext)
            self.logger.info(f"flat mode: {len(pairs)} matched (y, s) pairs")

        # ── patch 모드: land mask + 채택 patch 좌표를 한 번만 계산 (모든 날짜 공통) ──
        patch_mode = img_mode and args.patch is not None
        coords = None
        if patch_mode:
            P = int(args.patch)
            St = int(args.patch_stride) if args.patch_stride else P
            ref_path = next((yp for d, yp, _ in pairs if d == args.land_ref_date), pairs[0][1])
            land_mask = load_raw_bef32(ref_path, shape) > 0
            coords = self._land_patch_coords(land_mask, P, St, args.land_min)
            n_total = len(_patch_offsets(shape[0], P, St)) * len(_patch_offsets(shape[1], P, St))
            self.logger.info(f"patch 모드: p={P} stride={St} land_ref={ref_path.parent.name} "
                             f"land_min={args.land_min} → 채택 {len(coords)}/{n_total} patch/scene "
                             f"(land {land_mask.mean()*100:.1f}%)")
            if not coords:
                raise RuntimeError(f"land≥{args.land_min} patch 가 0개. land_min/patch 확인")
            # stacked 출력 디렉토리 + 좌표 기록 (s 는 공유라 1회만 저장)
            (out_dir / "y_patches").mkdir(parents=True, exist_ok=True)
            (out_dir / "cs_patches").mkdir(parents=True, exist_ok=True)
            with open(out_dir / "coords.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patch_idx", "y0", "x0", "patch", "stride", "land_ref", "land_min"])
                for k, (y0, x0) in enumerate(coords):
                    w.writerow([k, y0, x0, P, St, ref_path.parent.name, args.land_min])
            s_saved = False
        else:
            scenes_dir.mkdir(parents=True, exist_ok=True)

        scene_ids: List[str] = []
        for i, (sid, yp, sp) in enumerate(pairs):
            if img_mode:
                y = load_raw_bef32(yp, shape)
                s = s_cache
            else:
                y = load_image(yp)
                s = load_image(sp)
            if y.shape != s.shape:
                self.logger.warning(f"shape mismatch for {sid}: y={y.shape} s={s.shape} — skip")
                continue
            if multilook:
                # patch 모드는 patch 별로 계산(아래). full-scene 모드만 여기서 계산.
                cs = None if patch_mode else build_ratio_cs_multilook_numpy(
                    y, s, looks=args.looks, eps=args.eps, pol=args.pol)
            elif img_mode:
                cs = build_ratio_cs_patched_numpy(
                    y, s, ratio_denoiser=ratio_model,
                    patch_size=args.ratio_patch, stride=args.ratio_stride,
                    eps=args.eps, pol=args.pol, device=args.device,
                )
            else:
                cs = build_ratio_cs_numpy(y, s, ratio_denoiser=ratio_model,
                                          eps=args.eps, pol=args.pol, device=args.device)

            if patch_mode:
                # 한 날짜의 모든 land patch 를 묶어 저장. y 는 (N, P, P);
                # cs 는 denoiser면 (N, P, P), multilook 이면 (N, P/looks, P/looks).
                y_stack = np.stack([y[y0:y0 + P, x0:x0 + P] for (y0, x0) in coords]).astype(np.float32)
                if multilook:
                    cs_stack = np.stack([
                        build_ratio_cs_multilook_numpy(
                            y[y0:y0 + P, x0:x0 + P], s[y0:y0 + P, x0:x0 + P],
                            looks=args.looks, eps=args.eps, pol=args.pol)
                        for (y0, x0) in coords]).astype(np.float32)
                else:
                    cs_stack = np.stack([cs[y0:y0 + P, x0:x0 + P] for (y0, x0) in coords]).astype(np.float32)
                save_npy(out_dir / "y_patches" / f"{sid}.npy", y_stack)
                save_npy(out_dir / "cs_patches" / f"{sid}.npy", cs_stack)
                if not s_saved:  # s 는 모든 날짜 공통 → 1회만
                    s_stack = np.stack([s[y0:y0 + P, x0:x0 + P] for (y0, x0) in coords]).astype(np.float32)
                    save_npy(out_dir / "s_patches.npy", s_stack)
                    s_saved = True
                scene_ids.extend(f"{sid}:{k}" for k in range(len(coords)))
                self.logger.info(f"  prepared {i+1}/{len(pairs)}  ({sid}) → {len(coords)} patches "
                                 f"(y_patches/{sid}.npy {y_stack.shape})")
            else:
                d = scenes_dir / sid
                d.mkdir(parents=True, exist_ok=True)
                save_npy(d / "y.npy", y)
                save_npy(d / "s.npy", s)
                save_npy(d / "cs.npy", cs)
                scene_ids.append(sid)
                self.logger.info(f"  prepared {i+1}/{len(pairs)}  ({sid})")

        # split
        rng = random.Random(args.seed)
        rng.shuffle(scene_ids)
        n_val = max(1, int(len(scene_ids) * args.val_ratio)) if scene_ids else 0
        val_ids = scene_ids[:n_val]
        train_ids = scene_ids[n_val:]
        (splits_dir / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
        (splits_dir / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))
        self.logger.info(f"split written: train={len(train_ids)} val={len(val_ids)}")
        self.logger.info(f"done. data_root={out_dir.resolve()}")
