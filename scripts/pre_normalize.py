"""
Raw linear ratio bundles → per-patch random VV/VH selection → normalized [0, 1] bundles.

Each (date, patch_idx) independently picks VV (C11) or VH (C22) with p=0.5 at save
time (fixed seed → reproducible). VH uses L=2.9, VV uses L=2.5, so the normalized
distributions of the two pols are histogram-matched (Δquantile ≤ 0.015).

Layout
------
  in : training_lareunion_ratio/{date}_p512_s400_{C11,C22}.npy
       (N_patches, 512, 512) float32 linear ratio
  out: training_lareunion_ratio_intensity_patches/{date}_p512_s400.npy
       (N_patches, 512, 512) float32 in [0, 1]
       linear = 1 (no change)  → norm = 0.5
       linear = 0 (ocean/mask) → norm = 0       ← matches existing convention
  manifest: training_lareunion_ratio_intensity_patches/_pol_manifest.csv
       date,patch_idx,pol  — which pol was chosen per patch
"""
import sys
import csv
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, '/media/sdb8TB/naraspace/nst-sar-renoise/src')
from nstsr.config.norm_config import linear_to_norm_ratio

RAW_DIR = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')
OUT_DIR = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio_C11_patches')
SEED    = 0

OUT_DIR.mkdir(exist_ok=True)
manifest_path = OUT_DIR / '_pol_manifest.csv'

dates = sorted({f.name.split('_')[0] for f in RAW_DIR.glob('*_p512_s400_C11.npy')})
print(f'normalizing {len(dates)} dates  →  {OUT_DIR}')

rng = np.random.default_rng(SEED)
manifest_rows = []
t0 = time.time()
total_bytes = 0

for di, date in enumerate(dates):
    out_path = OUT_DIR / f'{date}_p512_s400.npy'
    if out_path.exists():
        continue

    vv = np.load(RAW_DIR / f'{date}_p512_s400_C11.npy')
    vh = np.load(RAW_DIR / f'{date}_p512_s400_C22.npy')
    assert vv.shape == vh.shape, f'shape mismatch on {date}'
    N = vv.shape[0]

    pick_vh = rng.integers(0, 2, size=N).astype(bool)
    out = np.empty_like(vv, dtype=np.float32)

    vv_idx = np.where(~pick_vh)[0]
    vh_idx = np.where(pick_vh)[0]
    if vv_idx.size:
        a = vv[vv_idx]
        ocean = (a == 0)
        n = linear_to_norm_ratio(a, pol='vv').astype(np.float32)
        n[ocean] = 0.0
        out[vv_idx] = n
    if vh_idx.size:
        a = vh[vh_idx]
        ocean = (a == 0)
        n = linear_to_norm_ratio(a, pol='vh').astype(np.float32)
        n[ocean] = 0.0
        out[vh_idx] = n

    np.save(out_path, out)
    total_bytes += out.nbytes

    for i in range(N):
        manifest_rows.append((date, i, 'vh' if pick_vh[i] else 'vv'))

    if (di + 1) % 10 == 0 or di == len(dates) - 1:
        elapsed = time.time() - t0
        rate = total_bytes / elapsed / 1e6
        eta = elapsed / (di + 1) * (len(dates) - di - 1)
        n_vv = sum(1 for r in manifest_rows if r[2] == 'vv')
        n_vh = len(manifest_rows) - n_vv
        print(f'  date {di+1:3d}/{len(dates)} ({date})  '
              f'elapsed {elapsed:6.1f}s  rate {rate:5.0f} MB/s  eta {eta:5.0f}s  '
              f'vv/vh so far: {n_vv}/{n_vh}')

with manifest_path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['date', 'patch_idx', 'pol'])
    w.writerows(manifest_rows)

n_vv = sum(1 for r in manifest_rows if r[2] == 'vv')
n_vh = len(manifest_rows) - n_vv
print(f'\ndone. wrote {total_bytes/1e9:.1f} GB in {time.time()-t0:.0f}s')
print(f'pol split: vv={n_vv}  vh={n_vh}  (total {len(manifest_rows)} patches)')
print(f'manifest: {manifest_path}')
