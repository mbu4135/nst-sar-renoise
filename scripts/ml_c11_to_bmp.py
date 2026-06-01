"""
Render every /media/sda8TB/SR/data_ref/LR/ratio/ML/YYYYMMDD/C11.img as
C11.bmp in its own folder, using a single global vmin/vmax (1st/99th
percentile of all non-zero values in dB).

C11.img layout: big-endian float32, BSQ, shape (920, 810)
(range_looks=16, azimuth_looks=4 of the 12960×3680 main ratio scene).
Zero pixels (masked / below noise floor) render as black.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT  = Path('/media/sda8TB/SR/data_ref/LR/ratio/ML')
SHAPE = (920, 810)
DTYPE = '>f4'
EPS   = 1e-12

paths = sorted(ROOT.glob('*/C11.img'))
print(f'found {len(paths)} C11.img files')

# Pass 1: load all and convert positive values to dB.
arrs_db = []
valid_masks = []
for i, p in enumerate(paths):
    a = np.fromfile(p, dtype=DTYPE).reshape(SHAPE)
    m = a > 0
    db = np.full(SHAPE, np.nan, dtype=np.float32)
    db[m] = 10.0 * np.log10(a[m] + EPS)
    arrs_db.append(db)
    valid_masks.append(m)

# Global percentile across all valid dB values.
all_vals = np.concatenate([db[m] for db, m in zip(arrs_db, valid_masks)])
vmin, vmax = np.percentile(all_vals, [1.0, 99.0])
print(f'global vmin={vmin:.3f} dB, vmax={vmax:.3f} dB  (N={all_vals.size:,})')

# Pass 2: normalize and write BMPs.
scale = 255.0 / (vmax - vmin)
for p, db, m in zip(paths, arrs_db, valid_masks):
    img = np.zeros(SHAPE, dtype=np.uint8)
    clipped = np.clip(db[m], vmin, vmax)
    img[m] = np.round((clipped - vmin) * scale).astype(np.uint8)
    out = p.with_suffix('.bmp')
    Image.fromarray(img, mode='L').save(out, format='BMP')

print(f'wrote {len(paths)} BMP files')
