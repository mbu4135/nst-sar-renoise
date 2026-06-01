"""
Drop locations where any date has more than 20% ocean (land_frac < 0.80).

Pass 1 : scan ONE C11 bundle for per-location land fraction. Locations are
         the same across dates and the 0.016% orbit-related footprint
         variation does not shift any threshold decision.
Pass 2 : rewrite every {date}_p512_s400_{C11,C22,C12_mag}.npy to contain only
         the kept rows (axis 0).
Pass 3 : remap local_idx in locations.csv & pairs.csv, update manifest.json.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

DST          = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')
LAND_THRESH  = 0.80                          # patch must be ≥80% land at every date
CHANNELS     = ('C11', 'C22', 'C12_mag')

locs = pd.read_csv(DST / 'locations.csv')
pairs = pd.read_csv(DST / 'pairs.csv', dtype={'date_a': str, 'date_b': str})
N_OLD = len(locs)

# ── Pass 1: per-location land fraction from a single date ──────────────────
c11_files = sorted(DST.glob('*_p512_s400_C11.npy'))
ref = c11_files[0]
print(f'pass 1: using {ref.name} as land-frac reference')
a = np.load(ref, mmap_mode='r')
land_frac = (a > 0).mean(axis=(1, 2)).astype(np.float32)
locs['land_frac'] = land_frac
keep_mask = (land_frac >= LAND_THRESH)
N_NEW = int(keep_mask.sum())
print(f'\nkept {N_NEW}/{N_OLD} locations  (min_land_frac ≥ {LAND_THRESH})')
print(f'dropped: {N_OLD - N_NEW}')
if N_OLD != N_NEW:
    print(locs[~keep_mask][['local_idx','global_loc_idx','x_orig','y_orig','land_frac']]
          .to_string(index=False))

old_to_new = {old: new for new, old in enumerate(np.where(keep_mask)[0])}

# ── Pass 2: subset every bundle ─────────────────────────────────────────────
all_files = sorted(DST.glob('*_p512_s400_*.npy'))
print(f'\npass 2: rewriting {len(all_files)} bundles')
t0 = time.time()
total_bytes = 0
for i, f in enumerate(all_files):
    a = np.load(f)                            # full load (200 MB)
    a_new = a[keep_mask]                      # subset axis 0
    np.save(f, a_new)
    total_bytes += a_new.nbytes
    if (i + 1) % 30 == 0 or i + 1 == len(all_files):
        rate = total_bytes / (time.time() - t0) / 1e6
        print(f'  {i+1:3d}/{len(all_files)}  elapsed {time.time()-t0:6.1f}s  rate {rate:5.0f} MB/s')

# ── Pass 3: remap metadata ──────────────────────────────────────────────────
locs_new = locs[keep_mask].reset_index(drop=True)
locs_new['old_local_idx'] = locs_new['local_idx']
locs_new['local_idx'] = np.arange(N_NEW)
locs_new = locs_new[['local_idx', 'old_local_idx', 'global_loc_idx',
                      'x_orig', 'y_orig', 'land_frac']]
locs_new.to_csv(DST / 'locations.csv', index=False)
print(f'\nupdated locations.csv ({N_NEW} rows)')

before = len(pairs)
pairs_new = pairs[pairs.local_idx.isin(old_to_new)].copy()
pairs_new['local_idx'] = pairs_new['local_idx'].map(old_to_new)
pairs_new = pairs_new.sort_values(['local_idx', 'date_a', 'date_b']).reset_index(drop=True)
pairs_new.to_csv(DST / 'pairs.csv', index=False)
print(f'updated pairs.csv: {before:,} → {len(pairs_new):,} rows')

# Manifest
mf = DST / 'manifest.json'
j = json.loads(mf.read_text())
j['n_locations']      = N_NEW
j['n_locations_dropped'] = N_OLD - N_NEW
j['land_frac_min']    = LAND_THRESH
j['land_frac_rule']   = (
    f'kept locations where (C11 > 0) fraction ≥ {LAND_THRESH} for EVERY date'
)
j['n_accepted_pairs'] = int(len(pairs_new))
mf.write_text(json.dumps(j, indent=2))
print(f'\ntotal data rewritten: {total_bytes/1e9:.1f} GB')
