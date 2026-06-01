"""
디버그용 시각화 — log-normalized 영상을 PNG 로 저장.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_grid_png(path: str | Path, images: list, titles: list | None = None,
                  vmin: float = 0.0, vmax: float = 1.0, vranges: list | None = None):
    """
    images  : list of 2D np.ndarray (또는 [1,H,W] torch tensor) — log-normalized 가정.
    vranges : 패널별 (vmin, vmax) 리스트. 주면 패널마다 다른 stretch 적용
              (None 이면 전역 vmin/vmax). 시각화 stretch 용.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("save_grid_png requires matplotlib") from e

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(images):
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        if img.ndim == 3:
            img = img[0]
        vlo, vhi = (vranges[i] if vranges is not None else (vmin, vmax))
        axes[i].imshow(img, cmap="gray", vmin=vlo, vmax=vhi)
        axes[i].set_axis_off()
        if titles is not None:
            axes[i].set_title(titles[i])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(path), dpi=120)
    plt.close(fig)
