"""
Sanity-check the pair index by visualizing a handful of pairs.

Samples 4 'best' accepted pairs, 4 'borderline' accepted pairs, and 4
rejected pairs spanning different locations. For each pair we plot the
two ML patches side by side plus the log-ratio map.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = Path('/media/sda8TB/SR/data_ref/LR/ratio/ML')
PAIRS    = Path('/media/sdb8TB/naraspace/nst-sar-renoise/outputs/pairs/pairs_all.csv.gz')
OUT_DIR  = Path('/media/sdb8TB/naraspace/nst-sar-renoise/outputs/pairs/viz')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_ML = (920, 810)
DTYPE    = '>f4'
P_H, P_W = 128, 32

rng = np.random.default_rng(0)

df = pd.read_csv(PAIRS, dtype={'date_a': str, 'date_b': str})
acc = df[df.accepted].copy()
rej = df[~df.accepted].copy()
acc['score'] = acc.logratio_mean.abs() + acc.logratio_std  # smaller is better

# 4 best (across distinct locations)
best = (acc.sort_values('score')
           .groupby('loc_idx', sort=False).head(1)   # one per location
           .head(4))

# 4 borderline accepted (score near 90th percentile of accepted)
q_lo, q_hi = acc.score.quantile([0.85, 0.95])
border_pool = acc[(acc.score >= q_lo) & (acc.score <= q_hi)]
border = border_pool.sample(4, random_state=0)

# 4 clearly rejected (score in middle of rejected distribution)
rej['score'] = rej.logratio_mean.abs() + rej.logratio_std
r_lo, r_hi = rej.score.quantile([0.5, 0.6])
rej_pool = rej[(rej.score >= r_lo) & (rej.score <= r_hi)]
rej_pick = rej_pool.sample(4, random_state=0)

cache = {}
def load_ml(date):
    if date not in cache:
        p = ROOT / date / 'C11.img'
        cache[date] = np.fromfile(p, dtype=DTYPE).reshape(SHAPE_ML)
    return cache[date]

def patch_db(date, x_ml, y_ml):
    a = load_ml(date)[y_ml:y_ml + P_H, x_ml:x_ml + P_W]
    db = np.full_like(a, np.nan, dtype=np.float32)
    np.log10(a, out=db, where=(a > 0))
    return 10.0 * db  # dB

# Global vmin/vmax already known from earlier run
VMIN_DB, VMAX_DB = -3.393, 2.518

def render(pairs, title, fname):
    n = len(pairs)
    fig, axes = plt.subplots(n, 3, figsize=(7, 1.6 * n + 0.5),
                              gridspec_kw={'wspace': 0.05, 'hspace': 0.35})
    if n == 1:
        axes = axes[None, :]
    for ax_row, (_, r) in zip(axes, pairs.iterrows()):
        A = patch_db(r.date_a, int(r.x_ml), int(r.y_ml))
        B = patch_db(r.date_b, int(r.x_ml), int(r.y_ml))
        R = (A - B) / 10.0  # back to log10 ratio
        # transpose so the 128-tall axis is shown vertical for legibility
        kwa = dict(vmin=VMIN_DB, vmax=VMAX_DB, cmap='gray', aspect='auto')
        ax_row[0].imshow(A.T, **kwa)
        ax_row[1].imshow(B.T, **kwa)
        ax_row[2].imshow(R.T, vmin=-0.4, vmax=0.4, cmap='RdBu_r', aspect='auto')
        ax_row[0].set_title(f"{r.date_a}", fontsize=8)
        ax_row[1].set_title(f"{r.date_b}", fontsize=8)
        ax_row[2].set_title(f"log10(A/B)  μ={r.logratio_mean:+.3f} σ={r.logratio_std:.3f}",
                             fontsize=8)
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
        ax_row[0].set_ylabel(f'loc {int(r.loc_idx)}\n({int(r.x_orig)},{int(r.y_orig)})',
                              fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_DIR/fname}')

render(best,    'BEST accepted pairs (different locations)',  'best.png')
render(border,  'BORDERLINE accepted pairs (near P30 cut)',   'borderline.png')
render(rej_pick,'REJECTED pairs (mid distribution)',          'rejected.png')

# Also dump the 12 pair rows for reference
sel = pd.concat([best.assign(group='best'),
                 border.assign(group='border'),
                 rej_pick.assign(group='rejected')])[
    ['group','loc_idx','x_orig','y_orig','x_ml','y_ml',
     'date_a','date_b','logratio_mean','logratio_std','valid_frac']]
sel.to_csv(OUT_DIR / 'samples.csv', index=False)
print(f'wrote {OUT_DIR/"samples.csv"}')
