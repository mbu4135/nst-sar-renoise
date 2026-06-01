"""
Inspect the final filtered dataset: render a few actual 512×512 patch pairs
straight from the bundles.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DST     = Path('/media/sdb8TB/naraspace/nst-sar-filtering/data/training_lareunion_ratio')
OUT_DIR = Path('/media/sdb8TB/naraspace/nst-sar-renoise/outputs/pairs/viz')
OUT_DIR.mkdir(parents=True, exist_ok=True)

pairs = pd.read_csv(DST / 'pairs.csv', dtype={'date_a': str, 'date_b': str})
locs  = pd.read_csv(DST / 'locations.csv')
pairs = pairs.merge(locs[['local_idx', 'x_orig', 'y_orig']], on='local_idx')
print(f'pairs: {len(pairs):,}, locations: {len(locs)}')

pairs['score'] = pairs.logratio_mean.abs() + pairs.logratio_std

# 6 best (one per distinct location for diversity)
best = (pairs.sort_values('score')
              .groupby('local_idx', sort=False).head(1)
              .head(6))
# 6 borderline (near the upper end of accepted distribution)
q_lo, q_hi = pairs.score.quantile([0.85, 0.95])
border = pairs[(pairs.score >= q_lo) & (pairs.score <= q_hi)].sample(6, random_state=0)

cache = {}
def load(date, ch='C11'):
    k = (date, ch)
    if k not in cache:
        cache[k] = np.load(DST / f'{date}_p512_s400_{ch}.npy', mmap_mode='r')
    return cache[k]

# Common dB stretch for VV intensity
VMIN_DB, VMAX_DB = -10.0, 6.0   # ratio-domain dB (p~5-95 of the C11 values)

def to_db(a):
    out = np.full_like(a, np.nan, dtype=np.float32)
    np.log10(a, out=out, where=(a > 0))
    return 10.0 * out

def render(picks, title, fname):
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), squeeze=False,
                              gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
    for ax_row, (_, r) in zip(axes, picks.iterrows()):
        A = load(r.date_a)[int(r.local_idx)]
        B = load(r.date_b)[int(r.local_idx)]
        Ad, Bd = to_db(A), to_db(B)
        R = np.where((A > 0) & (B > 0), np.log10(np.where(B > 0, A / np.maximum(B, 1e-12), 1.0)), np.nan)

        ax_row[0].imshow(Ad, vmin=VMIN_DB, vmax=VMAX_DB, cmap='gray')
        ax_row[1].imshow(Bd, vmin=VMIN_DB, vmax=VMAX_DB, cmap='gray')
        im = ax_row[2].imshow(R, vmin=-0.4, vmax=0.4, cmap='RdBu_r')

        ax_row[0].set_title(f"A: {r.date_a}", fontsize=9)
        ax_row[1].set_title(f"B: {r.date_b}", fontsize=9)
        ax_row[2].set_title(f"log10(A/B)  μ={r.logratio_mean:+.3f} σ={r.logratio_std:.3f}",
                             fontsize=9)
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
        ax_row[0].set_ylabel(
            f'loc {int(r.local_idx)}\n({int(r.x_orig)},{int(r.y_orig)})',
            fontsize=8,
        )
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT_DIR / fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT_DIR / fname)

render(best,   'BEST accepted pairs (one per location, 512×512 native)',  'dataset_best.png')
render(border, 'BORDERLINE accepted pairs (~P90 score)',                  'dataset_border.png')

# Save metadata for the picks
out = pd.concat([best.assign(group='best'), border.assign(group='border')])[
    ['group','local_idx','x_orig','y_orig','date_a','date_b',
     'logratio_mean','logratio_std','valid_frac']]
out.to_csv(OUT_DIR / 'dataset_samples.csv', index=False)
print('wrote', OUT_DIR / 'dataset_samples.csv')
