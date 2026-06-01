"""
Build the date-pair index for similarity-based training pairs.

Patch grid on the ORIGINAL ratio scene (12960 wide × 3680 tall):
    patch  = 512×512, stride = 400 → 32 × 8 = 256 locations.

The same grid in ML coords (range_looks=16, azimuth_looks=4):
    patch  = 32 (rg) × 128 (az), stride = 25 (rg) × 100 (az).
All grid origins land on integer ML pixels, so ML patches are exact.

Per location l and date pair (i, j) we read the two ML patches and compute:
    r = log10(A_i / A_j)      on pixels valid in both (>0)
    valid_frac, mean(r), std(r)

Patches with valid_frac < 0.5 are dropped (mostly-masked).

Acceptance flag (global percentile cut on the survivors):
    |mean(r)| < P30(|mean|)  AND  std(r) < P30(std)
i.e. roughly the 9 % of pairs that are most stationary in both bias and
speckle-baseline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time

ROOT      = Path('/media/sda8TB/SR/data_ref/LR/ratio/ML')
OUT_DIR   = Path('/media/sdb8TB/naraspace/nst-sar-renoise/outputs/pairs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_ML  = (920, 810)        # (lines/az, samples/rg)
DTYPE     = '>f4'

ORIG_H, ORIG_W = 3680, 12960
PATCH     = 512
STRIDE    = 400
RG_LOOKS  = 16                # ML range  downsample
AZ_LOOKS  = 4                 # ML azimuth downsample

P_W = PATCH // RG_LOOKS       # 32
P_H = PATCH // AZ_LOOKS       # 128
S_W = STRIDE // RG_LOOKS      # 25
S_H = STRIDE // AZ_LOOKS      # 100

VALID_FRAC_MIN = 0.5
PERCENTILE_CUT = 30.0

# ── 1. Build patch grid ──────────────────────────────────────────────────────
xs_orig = list(range(0, ORIG_W - PATCH + 1, STRIDE))   # 32 positions
ys_orig = list(range(0, ORIG_H - PATCH + 1, STRIDE))   # 8 positions
assert all(x % RG_LOOKS == 0 for x in xs_orig), 'range grid must align to looks'
assert all(y % AZ_LOOKS == 0 for y in ys_orig), 'azimuth grid must align to looks'
locations = [
    (yi * len(xs_orig) + xi, x, y, x // RG_LOOKS, y // AZ_LOOKS)
    for yi, y in enumerate(ys_orig)
    for xi, x in enumerate(xs_orig)
]
print(f'grid: {len(xs_orig)} × {len(ys_orig)} = {len(locations)} locations')
print(f'ML patch: {P_H}×{P_W} (az×rg), ML stride: {S_H}×{S_W}')

# ── 2. Load all ML scenes ────────────────────────────────────────────────────
date_dirs = sorted(d for d in ROOT.iterdir() if d.is_dir() and (d / 'C11.img').exists())
dates = [d.name for d in date_dirs]
N = len(dates)
t0 = time.time()
imgs = np.stack([np.fromfile(d / 'C11.img', dtype=DTYPE).reshape(SHAPE_ML)
                 for d in date_dirs]).astype(np.float32)
print(f'loaded {N} scenes → {imgs.shape}, {imgs.nbytes / 1e6:.0f} MB  ({time.time()-t0:.1f}s)')

# log10 with zero-mask (zeros stay NaN so they are excluded pairwise)
log_imgs = np.full_like(imgs, np.nan)
np.log10(imgs, out=log_imgs, where=(imgs > 0))

# ── 3. Pairwise log-ratio stats per location ─────────────────────────────────
records = []
t0 = time.time()
for loc_idx, x_o, y_o, x_ml, y_ml in locations:
    # (N, P_H, P_W) → (N, P)
    patch = log_imgs[:, y_ml:y_ml + P_H, x_ml:x_ml + P_W].reshape(N, -1)
    P = patch.shape[1]
    finite = np.isfinite(patch)

    for i in range(N - 1):
        ai = patch[i]                              # (P,)
        fi = finite[i]                             # (P,)
        rest = patch[i + 1:]                       # (M, P)
        rest_f = finite[i + 1:]                    # (M, P)
        mask = fi & rest_f                         # (M, P)
        n = mask.sum(axis=1).astype(np.float32)    # (M,)
        vf = n / P

        # safe diff (zero where invalid)
        diff = np.where(mask, ai - rest, 0.0)
        n_safe = np.maximum(n, 1.0)
        mean = diff.sum(axis=1) / n_safe
        sq   = np.where(mask, (ai - rest - mean[:, None]) ** 2, 0.0)
        std  = np.sqrt(sq.sum(axis=1) / n_safe)

        keep = vf >= VALID_FRAC_MIN
        if not keep.any():
            continue
        kk = np.where(keep)[0]
        for k in kk:
            j = i + 1 + int(k)
            records.append((loc_idx, x_o, y_o, x_ml, y_ml,
                            dates[i], dates[j],
                            float(mean[k]), float(std[k]), float(vf[k])))
    if (loc_idx + 1) % 32 == 0:
        print(f'  loc {loc_idx + 1}/{len(locations)} done  ({time.time()-t0:.1f}s, {len(records):,} pairs)')

df = pd.DataFrame.from_records(
    records,
    columns=['loc_idx', 'x_orig', 'y_orig', 'x_ml', 'y_ml',
             'date_a', 'date_b', 'logratio_mean', 'logratio_std', 'valid_frac'],
)
print(f'pairs with valid_frac ≥ {VALID_FRAC_MIN}: {len(df):,}  ({time.time()-t0:.1f}s)')

# ── 4. Percentile cut ────────────────────────────────────────────────────────
abs_mean = df['logratio_mean'].abs().to_numpy()
std_arr  = df['logratio_std'].to_numpy()
t_mean = np.percentile(abs_mean, PERCENTILE_CUT)
t_std  = np.percentile(std_arr,  PERCENTILE_CUT)
df['accepted'] = (abs_mean < t_mean) & (std_arr < t_std)
print(f'P{PERCENTILE_CUT:.0f} thresholds: |mean| < {t_mean:.4f}, std < {t_std:.4f}')
print(f'accepted pairs: {df.accepted.sum():,}  ({df.accepted.mean()*100:.1f}%)')

# ── 5. Save ──────────────────────────────────────────────────────────────────
out_full = OUT_DIR / 'pairs_all.csv.gz'
out_acc  = OUT_DIR / 'pairs_accepted.csv'
df.to_csv(out_full, index=False, compression='gzip')
df[df.accepted].drop(columns=['accepted']).to_csv(out_acc, index=False)
print(f'wrote {out_full}  ({out_full.stat().st_size/1e6:.1f} MB)')
print(f'wrote {out_acc}  ({out_acc.stat().st_size/1e6:.1f} MB)')

# Per-location summary
loc_summary = (df.groupby('loc_idx')
                 .agg(n_pairs=('accepted', 'size'),
                      n_accepted=('accepted', 'sum'),
                      median_mean=('logratio_mean', lambda s: float(s.abs().median())),
                      median_std=('logratio_std', 'median'))
                 .reset_index())
loc_summary.to_csv(OUT_DIR / 'loc_summary.csv', index=False)
print(f'wrote {OUT_DIR/"loc_summary.csv"}')
