"""
SARTripletDataset — (y, s, cs) triple 을 log-normalized [0, 1] 도메인에서 반환.

두 가지 데이터 레이아웃을 자동 감지한다.

(1) stacked (live, prepare --patch 출력) — 한 날짜의 land patch 를 묶어 저장:
    <root>/
        y_patches/<date>.npy     # (N, P, P) linear single-look
        cs_patches/<date>.npy    # [0,1] normalized ratio. 기본 multilook: (N, P/looks, P/looks)
                                 #   저해상 그대로 사용(upsample 없음) — augment 가 ratio 배수 offset 으로
                                 #   정렬 crop, 모델 cs_enc 가 bottleneck 해상도로 resize. denoiser 모드면 (N, P, P).
        s_patches.npy            # (N, P, P) linear temporal-avg (모든 날짜 공통, 1개)
        coords.csv               # patch_idx, y0, x0, ...
        splits/{train,val}.txt   # "date:idx" per line
    s 는 단일 공유 평균이므로 한 번만 저장되고 모든 날짜가 같은 patch index 로 공유.

(2) scene (prepare 기본/flat 출력):
    <root>/
        scenes/<scene_id>/{y.npy, s.npy, cs.npy}
        splits/{train,val}.txt   # scene_id per line
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from nstsr.config.norm_config import (
    normalize_image, normalize_r, normalize_mu_intensity, normalize_da,
)
from nstsr.data.ratio_builder import build_ratio_cs
from nstsr.data.transforms import (
    augment_triplet,
    center_crop_triplet,
    random_crop, center_crop, random_hflip, random_vflip,
    to_log10,
    to_tensor_2d,
)


class SARTripletDataset(Dataset):
    """
    한 sample = (y_norm, s_norm, cs_norm), 모두 log-normalized [0, 1], shape [1, H, W].

    mode (scene 레이아웃 전용):
        "cache_cs"  — 디스크에 미리 저장된 cs.npy 를 로드 (빠름).
        "online_cs" — 매 iter 마다 ratio_denoiser 로 즉석 계산.
    stacked 레이아웃은 항상 미리 계산된 cs_patches 를 사용 (cache).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        pol: str = "vv",
        patch_size: int = 128,
        mode: str = "cache_cs",
        ratio_denoiser: Optional[torch.nn.Module] = None,
        augment: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.root = Path(root)
        self.pol = pol
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment
        self.eps = eps

        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            raise RuntimeError(f"empty split: {split_file}")

        # ── 레이아웃 자동 감지 ────────────────────────────────────────────
        self.stacked = (self.root / "y_patches").is_dir()

        if self.stacked:
            self.samples = []
            for line in ids:
                date, k = line.split(":")
                self.samples.append((date, int(k)))
            self._y_cache: dict[str, np.ndarray] = {}
            self._cs_cache: dict[str, np.ndarray] = {}
            self._s = np.load(self.root / "s_patches.npy", mmap_mode="r")
        else:
            self.scene_ids = ids
            if mode == "online_cs":
                if ratio_denoiser is None:
                    raise ValueError("online_cs mode requires ratio_denoiser")
                for p in ratio_denoiser.parameters():
                    p.requires_grad_(False)
                ratio_denoiser.eval()
            self.ratio_denoiser = ratio_denoiser

    # ── helpers ─────────────────────────────────────────────────────────
    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / "scenes" / scene_id

    def _load_image_norm(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        return to_tensor_2d(normalize_image(to_log10(arr, eps=self.eps), pol=self.pol))

    def _load_cs_norm(self, path: Path) -> torch.Tensor:
        return to_tensor_2d(np.load(path).astype(np.float32))

    def _mmap(self, cache: dict, sub: str, date: str) -> np.ndarray:
        if date not in cache:
            cache[date] = np.load(self.root / sub / f"{date}.npy", mmap_mode="r")
        return cache[date]

    def _norm_from_linear(self, arr_linear: np.ndarray) -> torch.Tensor:
        a = np.asarray(arr_linear, dtype=np.float32)
        return to_tensor_2d(normalize_image(to_log10(a, eps=self.eps), pol=self.pol))

    # ── Dataset protocol ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples) if self.stacked else len(self.scene_ids)

    def __getitem__(self, idx: int):
        if self.stacked:
            date, k = self.samples[idx]
            y = self._norm_from_linear(self._mmap(self._y_cache, "y_patches", date)[k])
            s = self._norm_from_linear(self._s[k])
            # multilook cs 는 patch/looks 해상도(저해상) 그대로 사용. upsample 없이
            # augment/center crop 에서 ratio 배수 offset 으로 정렬 crop, 모델 내부
            # cs_enc 가 bottleneck 해상도로 resize. (MCAM cs 를 bottleneck 에 1회 주입)
            cs = to_tensor_2d(np.array(self._mmap(self._cs_cache, "cs_patches", date)[k],
                                       dtype=np.float32))  # copy: mmap 은 read-only
            sid = f"{date}:{k}"
        else:
            scene_id = self.scene_ids[idx]
            d = self._scene_dir(scene_id)
            y = self._load_image_norm(d / "y.npy")
            s = self._load_image_norm(d / "s.npy")
            if self.mode == "cache_cs":
                cs = self._load_cs_norm(d / "cs.npy")
            else:  # online_cs
                y_lin = np.load(d / "y.npy").astype(np.float32)
                s_lin = np.load(d / "s.npy").astype(np.float32)
                y_lin_t = torch.from_numpy(y_lin).unsqueeze(0).unsqueeze(0)
                s_lin_t = torch.from_numpy(s_lin).unsqueeze(0).unsqueeze(0)
                cs = build_ratio_cs(y_lin_t, s_lin_t, ratio_denoiser=self.ratio_denoiser,
                                    eps=self.eps, pol=self.pol).squeeze(0)
            sid = scene_id

        if self.augment:
            y, s, cs = augment_triplet(y, s, cs, patch=self.patch_size)
        else:
            y, s, cs = center_crop_triplet(y, s, cs, self.patch_size)

        return {"y": y, "s": s, "cs": cs, "scene_id": sid}


# ─────────────────────────────────────────────────────────────────────────────
# renoise speckle 설계 (현행)
# ─────────────────────────────────────────────────────────────────────────────

class SARSpeckleDataset(Dataset):
    """
    speckle_ds 로더. target r = log10(y_t/μ) (=speckle), conditioning = (μ, D_A, valid).

    sample id = "date:k"  (k = 그 날짜 r_patches 내 로컬 인덱스).
    layout (build_speckle_ds.py 출력):
        <root>/
            mu_patches.npy, da_patches.npy, valid_patches.npy   # (N, P, P) 공유 — coords idx 로 매핑
            r_patches/<date>.npy        # (Nk, P, P) raw log10(y/μ)  (정규화는 여기서)
            cmask_patches/<date>.npy    # (Nk, P, P) uint8 loss mask (shadow/change/no-data=0)
            idx_patches/<date>.npy      # (Nk,) 공유 cond 의 coords idx
            splits/{train,val}.txt      # "date:k" per line

    반환 (모두 [C, h, w] tensor, augment 후 patch_size):
        r     : [1, h, w]  normalize_r(log10 ratio) ∈ [-1, 1]   (diffusion x0/target)
        cond  : [3, h, w]  [μ_norm, D_A_norm, valid]            (model conditioning)
        cmask : [1, h, w]  loss mask ∈ {0, 1}
    """

    def __init__(self, root, split: str = "train", patch_size: int = 128, augment: bool = True):
        super().__init__()
        self.root = Path(root)
        self.patch_size = patch_size
        self.augment = augment

        split_file = self.root / "splits" / f"{split}.txt"
        with open(split_file) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        if not ids:
            raise RuntimeError(f"empty split: {split_file}")
        self.samples = [(s.split(":")[0], int(s.split(":")[1])) for s in ids]

        self._mu = np.load(self.root / "mu_patches.npy", mmap_mode="r")
        self._da = np.load(self.root / "da_patches.npy", mmap_mode="r")
        self._val = np.load(self.root / "valid_patches.npy", mmap_mode="r")
        self._r_cache: dict[str, np.ndarray] = {}
        self._cm_cache: dict[str, np.ndarray] = {}
        self._idx_cache: dict[str, np.ndarray] = {}

    def _mmap(self, cache: dict, sub: str, date: str) -> np.ndarray:
        if date not in cache:
            cache[date] = np.load(self.root / sub / f"{date}.npy", mmap_mode="r")
        return cache[date]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        date, k = self.samples[idx]
        r_raw = np.array(self._mmap(self._r_cache, "r_patches", date)[k], dtype=np.float32)
        cm = np.array(self._mmap(self._cm_cache, "cmask_patches", date)[k], dtype=np.float32)
        gidx = int(self._mmap(self._idx_cache, "idx_patches", date)[k])
        mu = np.asarray(self._mu[gidx], dtype=np.float32)
        da = np.asarray(self._da[gidx], dtype=np.float32)
        val = np.asarray(self._val[gidx], dtype=np.float32)

        r = to_tensor_2d(normalize_r(r_raw))                 # [1,P,P] ∈ [-1,1]
        mu_n = to_tensor_2d(normalize_mu_intensity(mu))      # [1,P,P]
        da_n = to_tensor_2d(normalize_da(da))                # [1,P,P]
        valt = to_tensor_2d(val)                             # [1,P,P]
        cmask = to_tensor_2d(cm)                             # [1,P,P]
        cond = torch.cat([mu_n, da_n, valt], dim=0)          # [3,P,P]

        if self.augment:
            r, cond, cmask = random_crop((r, cond, cmask), self.patch_size)
            r, cond, cmask = random_hflip((r, cond, cmask))
            r, cond, cmask = random_vflip((r, cond, cmask))
        else:
            r, cond, cmask = center_crop((r, cond, cmask), self.patch_size)

        return {"r": r, "cond": cond, "cmask": cmask, "scene_id": f"{date}:{k}"}
