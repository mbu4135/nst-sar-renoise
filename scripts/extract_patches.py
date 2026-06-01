"""
Extract original-resolution 512×512 patches for every accepted-pair location
and bundle them per-date.

Source : /media/sda8TB/SR/data_ref/LR/ratio/{date}/{C11,C22,C12_mag}.img
         big-endian float32, BSQ, shape (3680, 12960).
Output : /media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio/
         ├── {date}_p512_s400_C11.npy       (N_loc, 512, 512) float32 LE
         ├── {date}_p512_s400_C22.npy       same shape
         ├── {date}_p512_s400_C12_mag.npy   same shape
         ├── locations.csv      local_idx, global_loc_idx, x_orig, y_orig
         ├── pairs.csv          date_a, date_b, local_idx, logratio_mean, ...
         ├── dates.txt          ordered list of dates
         └── manifest.json      patch size, stride, channels, counts

`local_idx` (0..N-1) indexes the location axis in every .npy bundle and is
the join key in pairs.csv. Only locations that appear in at least one
accepted ML pair are kept (the rest are fully masked / never useful).
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

SRC      = Path('/media/sda8TB/SR/data_ref/LR/ratio')
DST      = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')
PAIRS_IN = Path('/media/sdb8TB/naraspace/nst-sar-renoise/outputs/pairs/pairs_all.csv.gz')
DST.mkdir(parents=True, exist_ok=True)

SHAPE_ORIG = (3680, 12960)
DTYPE      = '>f4'
PATCH      = 512
STRIDE     = 400
CHANNELS   = ('C11', 'C22', 'C12_mag')
ORIG_H, ORIG_W = SHAPE_ORIG

# ── 1. Rebuild the patch grid (must match build_pair_index.py exactly) ───────
xs = list(range(0, ORIG_W - PATCH + 1, STRIDE))   # 32
ys = list(range(0, ORIG_H - PATCH + 1, STRIDE))   # 8
grid = [(yi * len(xs) + xi, x, y)
        for yi, y in enumerate(ys)
        for xi, x in enumerate(xs)]
assert len(grid) == 256

# ── 2. Determine kept locations from accepted pairs ──────────────────────────
df = pd.read_csv(PAIRS_IN, dtype={'date_a': str, 'date_b': str})
acc = df[df.accepted].copy()
kept_global = sorted(acc.loc_idx.unique().tolist())
global2local = {g: i for i, g in enumerate(kept_global)}
acc['local_idx'] = acc.loc_idx.map(global2local)

locations = pd.DataFrame([
    (local, glob, grid[glob][1], grid[glob][2])
    for glob, local in global2local.items()
], columns=['local_idx', 'global_loc_idx', 'x_orig', 'y_orig']).sort_values('local_idx').reset_index(drop=True)
N_LOC = len(locations)
print(f'kept {N_LOC}/256 locations (those participating in ≥1 accepted pair)')

dates = sorted({d.name for d in SRC.iterdir() if d.is_dir() and d.name[:4].isdigit()
                and all((d / f'{c}.img').exists() for c in CHANNELS)})
print(f'dates: {len(dates)}  ({dates[0]} → {dates[-1]})')

# ── 3. Write metadata up front (fail fast if path wrong) ─────────────────────
locations.to_csv(DST / 'locations.csv', index=False)
(DST / 'dates.txt').write_text('\n'.join(dates) + '\n')
acc_out = acc[['date_a', 'date_b', 'local_idx',
               'logratio_mean', 'logratio_std', 'valid_frac']].sort_values(
                   ['local_idx', 'date_a', 'date_b']).reset_index(drop=True)
acc_out.to_csv(DST / 'pairs.csv', index=False)
manifest = {
    'patch_size': PATCH,
    'stride': STRIDE,
    'channels': list(CHANNELS),
    'shape_orig': list(SHAPE_ORIG),
    'dtype_src': DTYPE,
    'dtype_dst': '<f4',
    'n_locations': N_LOC,
    'n_dates': len(dates),
    'n_accepted_pairs': int(len(acc_out)),
    'pair_source': str(PAIRS_IN),
}
(DST / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('wrote metadata: locations.csv, dates.txt, pairs.csv, manifest.json')

# ── 4. Extract & save patches ────────────────────────────────────────────────
xy_local = [(grid[g][1], grid[g][2]) for g in kept_global]
total_bytes = 0
t0 = time.time()
for di, date in enumerate(dates):
    for ch in CHANNELS:
        out_path = DST / f'{date}_p512_s400_{ch}.npy'
        if out_path.exists():
            continue
        img = np.fromfile(SRC / date / f'{ch}.img', dtype=DTYPE).reshape(SHAPE_ORIG)
        # Stack patches; cast to native little-endian float32
        patches = np.empty((N_LOC, PATCH, PATCH), dtype=np.float32)
        for k, (x, y) in enumerate(xy_local):
            patches[k] = img[y:y + PATCH, x:x + PATCH]
        np.save(out_path, patches)
        total_bytes += patches.nbytes
    if (di + 1) % 10 == 0 or di == len(dates) - 1:
        elapsed = time.time() - t0
        rate = total_bytes / elapsed / 1e6
        eta = elapsed / (di + 1) * (len(dates) - di - 1)
        print(f'  date {di + 1:3d}/{len(dates)} ({date})  '
              f'elapsed {elapsed:6.1f}s  rate {rate:5.0f} MB/s  eta {eta:5.0f}s')

print(f'\nDone. total written ≈ {total_bytes/1e9:.1f} GB in {time.time()-t0:.0f}s')
