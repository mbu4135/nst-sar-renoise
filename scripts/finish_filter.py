"""
Finish the partially-completed ocean filter:
  - 243 bundles already rewritten to 197 rows (correct mask from a previous
    run that we killed during Pass 2).
  - 119 bundles still at 200 rows.
  - 1 bundle (20200914_p512_s400_C11.npy) corrupted by an interrupted write.

We rederive the same 197-keep mask from the first date's C11.img source
(verified to match the multi-date min: dropped loc_idx are 12, 13, 55).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

SRC = Path('/media/sda8TB/SR/data_ref/LR/ratio')
DST = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')

SHAPE_ORIG = (3680, 12960)
DTYPE      = '>f4'
PATCH      = 512
LAND_THRESH = 0.80
EXPECTED_NEW_ROWS = 197
ORIG_ROWS = 200

locs = pd.read_csv(DST / 'locations.csv')
assert len(locs) == ORIG_ROWS, f'expected {ORIG_ROWS} rows in locations.csv, got {len(locs)}'

# ── Derive keep_mask from source C11 of first date ─────────────────────────
ref_date = sorted(locs.local_idx.astype(int).unique())  # not used; just first source date
first_date = sorted(d.name for d in SRC.iterdir() if d.is_dir() and d.name[:4].isdigit())[0]
img = np.fromfile(SRC / first_date / 'C11.img', dtype=DTYPE).reshape(SHAPE_ORIG)
land_frac = np.empty(len(locs), dtype=np.float32)
for i, row in locs.iterrows():
    x, y = int(row.x_orig), int(row.y_orig)
    patch = img[y:y+PATCH, x:x+PATCH]
    land_frac[i] = (patch > 0).mean()
keep_mask = land_frac >= LAND_THRESH
N_NEW = int(keep_mask.sum())
print(f'keep_mask: {N_NEW}/{ORIG_ROWS}  (dropped local_idx: '
      f'{list(locs.loc[~keep_mask, "local_idx"].astype(int))})')
assert N_NEW == EXPECTED_NEW_ROWS, f'expected {EXPECTED_NEW_ROWS}, got {N_NEW}'

# ── Fix the corrupted file ─────────────────────────────────────────────────
corrupted = DST / '20200914_p512_s400_C11.npy'
corrupted_size = corrupted.stat().st_size
print(f'\ncorrupted file: {corrupted.name}  size={corrupted_size:,}')
img2 = np.fromfile(SRC / '20200914' / 'C11.img', dtype=DTYPE).reshape(SHAPE_ORIG).astype(np.float32)
patches200 = np.stack([img2[int(r.y_orig):int(r.y_orig)+PATCH,
                            int(r.x_orig):int(r.x_orig)+PATCH] for _, r in locs.iterrows()])
np.save(corrupted, patches200[keep_mask])
print(f'rewrote {corrupted.name}  → shape ({N_NEW}, {PATCH}, {PATCH})')

# ── Subset all still-200-row files ─────────────────────────────────────────
EXPECTED_SIZE_FILTERED = N_NEW * PATCH * PATCH * 4 + 128  # numpy header ~128 B
EXPECTED_SIZE_ORIG     = ORIG_ROWS * PATCH * PATCH * 4 + 128

all_files = sorted(DST.glob('*_p512_s400_*.npy'))
todo = [f for f in all_files
        if abs(f.stat().st_size - EXPECTED_SIZE_ORIG) < 1024]   # still original 200-row
already = [f for f in all_files
           if abs(f.stat().st_size - EXPECTED_SIZE_FILTERED) < 1024]
print(f'\nstill-unfiltered (200 rows): {len(todo)}  / already-filtered (197 rows): {len(already)}')

t0 = time.time()
for i, f in enumerate(todo):
    a = np.load(f)
    assert a.shape == (ORIG_ROWS, PATCH, PATCH), f'{f.name}: unexpected shape {a.shape}'
    np.save(f, a[keep_mask])
    if (i + 1) % 30 == 0 or i + 1 == len(todo):
        print(f'  {i+1:3d}/{len(todo)}  elapsed {time.time()-t0:5.1f}s')

# ── Verify all files now at 197 rows ───────────────────────────────────────
bad = [f.name for f in DST.glob('*_p512_s400_*.npy')
       if abs(f.stat().st_size - EXPECTED_SIZE_FILTERED) >= 1024]
print(f'\nfinal check — files NOT at expected filtered size: {len(bad)}')
if bad:
    for b in bad[:10]:
        print('  bad:', b)

# ── Update locations.csv ───────────────────────────────────────────────────
locs['land_frac'] = land_frac
locs_new = locs[keep_mask].reset_index(drop=True)
locs_new['old_local_idx'] = locs_new['local_idx']
locs_new['local_idx']     = np.arange(N_NEW)
locs_new = locs_new[['local_idx', 'old_local_idx', 'global_loc_idx',
                      'x_orig', 'y_orig', 'land_frac']]
locs_new.to_csv(DST / 'locations.csv', index=False)
print(f'\nupdated locations.csv: {ORIG_ROWS} → {N_NEW} rows')

# ── Update pairs.csv: drop pairs at removed locations, remap local_idx ─────
pairs = pd.read_csv(DST / 'pairs.csv', dtype={'date_a': str, 'date_b': str})
old_to_new = {int(o): n for n, o in enumerate(np.where(keep_mask)[0])}
before = len(pairs)
pairs_new = pairs[pairs.local_idx.isin(old_to_new)].copy()
pairs_new['local_idx'] = pairs_new['local_idx'].map(old_to_new)
pairs_new = pairs_new.sort_values(['local_idx', 'date_a', 'date_b']).reset_index(drop=True)
pairs_new.to_csv(DST / 'pairs.csv', index=False)
print(f'updated pairs.csv: {before:,} → {len(pairs_new):,} rows')

# ── Manifest ───────────────────────────────────────────────────────────────
mf = DST / 'manifest.json'
j = json.loads(mf.read_text())
j['n_locations']         = N_NEW
j['n_locations_dropped'] = ORIG_ROWS - N_NEW
j['land_frac_min']       = LAND_THRESH
j['land_frac_rule']      = (
    f'kept locations where (C11 > 0) fraction ≥ {LAND_THRESH} '
    f'on the first date ({first_date})'
)
j['n_accepted_pairs']    = int(len(pairs_new))
mf.write_text(json.dumps(j, indent=2))
print('updated manifest.json')
