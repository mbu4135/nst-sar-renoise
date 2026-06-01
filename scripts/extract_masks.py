"""
Derive per-date land masks from the already-extracted C11 patch bundles.

For each date we write {date}_p512_s400_mask.npy of shape (N_loc, 512, 512)
uint8, where 1 = land (C11 > 0), 0 = ocean / masked. Effective per-pair
mask for loss/normalization downstream is mask_a & mask_b.
"""
from pathlib import Path
import numpy as np
import json

DST = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')
c11_files = sorted(DST.glob('*_p512_s400_C11.npy'))
print(f'found {len(c11_files)} C11 bundles')

for f in c11_files:
    out = f.with_name(f.name.replace('_C11.npy', '_mask.npy'))
    if out.exists():
        continue
    a = np.load(f, mmap_mode='r')
    m = (a > 0).astype(np.uint8)
    np.save(out, m)
print('done')

# Update manifest if present
mf = DST / 'manifest.json'
if mf.exists():
    j = json.loads(mf.read_text())
    j['mask_dtype'] = 'uint8'
    j['mask_rule']  = 'C11 > 0 → 1 (land), 0 (ocean/masked)'
    j['channels_with_mask'] = list(j.get('channels', [])) + ['mask']
    mf.write_text(json.dumps(j, indent=2))
    print('updated manifest.json')
