"""
SAR 영상 I/O — .npy / .tif (tifffile) / .raw (big-endian float32) 지원.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ── 단순 .npy ────────────────────────────────────────────────────────────────

def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)


# ── tiff (tifffile 필요) ─────────────────────────────────────────────────────

def save_tiff(path: str | Path, arr: np.ndarray) -> None:
    try:
        import tifffile
    except ImportError as e:
        raise ImportError("save_tiff requires `tifffile` (pip install tifffile)") from e
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr.astype(np.float32, copy=False))


def load_tiff(path: str | Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as e:
        raise ImportError("load_tiff requires `tifffile` (pip install tifffile)") from e
    return tifffile.imread(str(path)).astype(np.float32, copy=False)


# ── big-endian float32 raw ───────────────────────────────────────────────────

def save_raw_bef32(path: str | Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.astype(">f4").tofile(str(path))


def load_raw_bef32(path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    return np.fromfile(str(path), dtype=">f4").reshape(shape).astype(np.float32)


# ── 일반화된 dispatch ────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """확장자에 따라 load_npy / load_tiff. raw 는 별도 load_raw_bef32 사용."""
    p = Path(path)
    if p.suffix.lower() in {".npy"}:
        return load_npy(p)
    if p.suffix.lower() in {".tif", ".tiff"}:
        return load_tiff(p)
    raise ValueError(f"unknown extension: {p.suffix}")


def save_image(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    if p.suffix.lower() in {".npy"}:
        return save_npy(p, arr)
    if p.suffix.lower() in {".tif", ".tiff"}:
        return save_tiff(p, arr)
    raise ValueError(f"unknown extension: {p.suffix}")
