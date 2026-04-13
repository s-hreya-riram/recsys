"""
Generates convergence plots (val NDCG@10 vs epoch) for NCF and Two-Tower
on both datasets. MF/ALS and Popularity performance shown as horizontal references.

Usage:
    python src/plot_convergence.py

Output:
    results/figures/convergence_movielens.png
    results/figures/convergence_amazonmusic.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

OUT_DIR = Path('results/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colours
NCF_COLOR = '#E15759'
TT_COLOR  = '#4E79A7'
MF_COLOR  = '#59A14F'
POP_COLOR = '#B07AA1'

# ── MovieLens ─────────────────────────────────────────────────────────────────
# NCF best config: emb=32, layers=[128,64,32], lr=0.001, n_neg=8
# From tuning log
'''
    NCF training on mps: emb=32, layers=[128, 64, 32], neg=8, lr=0.001
      epoch 1/50 train_loss=0.2647 val_NDCG@10=0.0481
      epoch 2/50 train_loss=0.2444 val_NDCG@10=0.0490
      epoch 3/50 train_loss=0.2380 val_NDCG@10=0.0534
      epoch 4/50 train_loss=0.2249 val_NDCG@10=0.0551
      epoch 5/50 train_loss=0.2044 val_NDCG@10=0.0531
      epoch 6/50 train_loss=0.1922 val_NDCG@10=0.0535
      epoch 7/50 train_loss=0.1819 val_NDCG@10=0.0540
      epoch 8/50 train_loss=0.1719 val_NDCG@10=0.0585
      epoch 9/50 train_loss=0.1642 val_NDCG@10=0.0576
      epoch 10/50 train_loss=0.1569 val_NDCG@10=0.0571
      epoch 11/50 train_loss=0.1524 val_NDCG@10=0.0581
      epoch 12/50 train_loss=0.1478 val_NDCG@10=0.0557
      epoch 13/50 train_loss=0.1436 val_NDCG@10=0.0556
      early stopping at epoch 13
      best val NDCG@10: 0.0585
'''
ncf_ml = [
    0.0481, 0.0490, 0.0534, 0.0551, 0.0531, 0.0535, 0.0540, 0.0585,
    0.0576, 0.0571, 0.0581, 0.0557, 0.0556
]
ncf_ml_best = 8   # epoch of best val NDCG (1-indexed)

# Two-Tower best config: emb=32, layers=[64], lr=0.001, n_neg=8
# From tuning log
'''
    two-tower [explicit] on mps: emb=32, layers=[64], neg=8, lr=0.001
      epoch 1/50 train_loss=0.3381 val_NDCG@10=0.0492
      epoch 2/50 train_loss=0.3144 val_NDCG@10=0.0462
      epoch 3/50 train_loss=0.2916 val_NDCG@10=0.0461
      epoch 4/50 train_loss=0.2510 val_NDCG@10=0.0538
      epoch 5/50 train_loss=0.2211 val_NDCG@10=0.0542
      epoch 6/50 train_loss=0.2065 val_NDCG@10=0.0553
      epoch 7/50 train_loss=0.1942 val_NDCG@10=0.0558
      epoch 8/50 train_loss=0.1820 val_NDCG@10=0.0537
      epoch 9/50 train_loss=0.1703 val_NDCG@10=0.0529
      epoch 10/50 train_loss=0.1633 val_NDCG@10=0.0574
      epoch 11/50 train_loss=0.1568 val_NDCG@10=0.0582
      epoch 12/50 train_loss=0.1503 val_NDCG@10=0.0564
      epoch 13/50 train_loss=0.1460 val_NDCG@10=0.0523
      epoch 14/50 train_loss=0.1415 val_NDCG@10=0.0588
      epoch 15/50 train_loss=0.1386 val_NDCG@10=0.0640
      epoch 16/50 train_loss=0.1352 val_NDCG@10=0.0621
      epoch 17/50 train_loss=0.1325 val_NDCG@10=0.0634
      epoch 18/50 train_loss=0.1309 val_NDCG@10=0.0681
      epoch 19/50 train_loss=0.1285 val_NDCG@10=0.0631
      epoch 20/50 train_loss=0.1263 val_NDCG@10=0.0601
      epoch 21/50 train_loss=0.1251 val_NDCG@10=0.0595
      epoch 22/50 train_loss=0.1233 val_NDCG@10=0.0604
      epoch 23/50 train_loss=0.1219 val_NDCG@10=0.0572
      early stopping at epoch 23
      best val NDCG@10: 0.0681
'''
tt_ml = [
    0.0492, 0.0462, 0.0461, 0.0538, 0.0542, 0.0553, 0.0558, 0.0537,
    0.0529, 0.0574, 0.0582, 0.0564, 0.0523, 0.0588, 0.0640, 0.0621,
    0.0634, 0.0681, 0.0631, 0.0601, 0.0595, 0.0604, 0.0572
]
tt_ml_best = 18

# Reference lines (val NDCG@10 from tuning)
svd_ml_val  = 0.0152   # SVD best: factors=50, reg=0.001, epochs=300
pop_ml_val  = 0.0366   # Popularity test NDCG used as proxy

# ── Amazon Music ──────────────────────────────────────────────────────────────
# NCF best config: emb=64, layers=[128,64,32], lr=0.001, n_neg=4
# From pipeline run log
'''
  Fitting ncf...
    NCF training on mps: emb=64, layers=[128, 64, 32], neg=4, lr=0.001
      epoch 1/50 train_loss=0.4595 val_NDCG@10=0.0065
      epoch 2/50 train_loss=0.4341 val_NDCG@10=0.0072
      epoch 3/50 train_loss=0.4154 val_NDCG@10=0.0086
      epoch 4/50 train_loss=0.3858 val_NDCG@10=0.0088
      epoch 5/50 train_loss=0.3542 val_NDCG@10=0.0094
      epoch 6/50 train_loss=0.3265 val_NDCG@10=0.0099
      epoch 7/50 train_loss=0.3023 val_NDCG@10=0.0099
      epoch 8/50 train_loss=0.2809 val_NDCG@10=0.0097
      epoch 9/50 train_loss=0.2620 val_NDCG@10=0.0082
      epoch 10/50 train_loss=0.2455 val_NDCG@10=0.0107
      epoch 11/50 train_loss=0.2296 val_NDCG@10=0.0098
      epoch 12/50 train_loss=0.2162 val_NDCG@10=0.0097
      epoch 13/50 train_loss=0.2051 val_NDCG@10=0.0086
      epoch 14/50 train_loss=0.1940 val_NDCG@10=0.0101
      epoch 15/50 train_loss=0.1849 val_NDCG@10=0.0106
      early stopping at epoch 15
      best val NDCG@10: 0.0107
'''
ncf_am = [
    0.0065, 0.0072, 0.0086, 0.0088, 0.0094, 0.0099,
    0.0099, 0.0097, 0.0082, 0.0107, 0.0098, 0.0097, 0.0086, 0.0101, 0.0106
]
ncf_am_best = 10

# Two-Tower best config: emb=64, layers=[64], lr=0.001, n_neg=8
# From tuning log (best performing config)
'''
  Fitting two_tower...
    two-tower [implicit] on mps: emb=64, layers=[64], neg=8, lr=0.001
      epoch 1/50 train_loss=0.4969 val_NDCG@10=0.0059
      epoch 2/50 train_loss=0.3719 val_NDCG@10=0.0065
      epoch 3/50 train_loss=0.2901 val_NDCG@10=0.0064
      epoch 4/50 train_loss=0.2449 val_NDCG@10=0.0081
      epoch 5/50 train_loss=0.2179 val_NDCG@10=0.0093
      epoch 6/50 train_loss=0.2033 val_NDCG@10=0.0106
      epoch 7/50 train_loss=0.1902 val_NDCG@10=0.0123
      epoch 8/50 train_loss=0.1796 val_NDCG@10=0.0117
      epoch 9/50 train_loss=0.1720 val_NDCG@10=0.0121
      epoch 10/50 train_loss=0.1660 val_NDCG@10=0.0136
      epoch 11/50 train_loss=0.1620 val_NDCG@10=0.0129
      epoch 12/50 train_loss=0.1577 val_NDCG@10=0.0138
      epoch 13/50 train_loss=0.1533 val_NDCG@10=0.0139
      epoch 14/50 train_loss=0.1507 val_NDCG@10=0.0143
      epoch 15/50 train_loss=0.1474 val_NDCG@10=0.0160
      epoch 16/50 train_loss=0.1460 val_NDCG@10=0.0150
      epoch 17/50 train_loss=0.1435 val_NDCG@10=0.0157
      epoch 18/50 train_loss=0.1407 val_NDCG@10=0.0161
      epoch 19/50 train_loss=0.1391 val_NDCG@10=0.0157
      epoch 20/50 train_loss=0.1381 val_NDCG@10=0.0155
      epoch 21/50 train_loss=0.1365 val_NDCG@10=0.0163
      epoch 22/50 train_loss=0.1350 val_NDCG@10=0.0159
      epoch 23/50 train_loss=0.1336 val_NDCG@10=0.0158
      epoch 24/50 train_loss=0.1332 val_NDCG@10=0.0159
      epoch 25/50 train_loss=0.1327 val_NDCG@10=0.0156
      epoch 26/50 train_loss=0.1310 val_NDCG@10=0.0160
      early stopping at epoch 26
      best val NDCG@10: 0.0163
'''
tt_am = [
    0.0059, 0.0065, 0.0064, 0.0081, 0.0093, 0.0106, 0.0123, 0.0117, 0.0121,
    0.0136, 0.0129, 0.0138, 0.0139, 0.0143, 0.0160, 0.0150, 0.0157, 0.0161,
    0.0157, 0.0155, 0.0163, 0.0159, 0.0158, 0.0159, 0.0156, 0.0160
]
tt_am_best = 21

# Reference lines
als_am_val = 0.0397   # ALS best val NDCG: factors=200, reg=0.01, epochs=30, alpha=100
pop_am_val = 0.0076   # Popularity test NDCG


def make_plot(ncf_data, tt_data, ncf_best, tt_best,
              ref_val, ref_label, pop_val,
              title, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F8F8')

    ne = list(range(1, len(ncf_data) + 1))
    te = list(range(1, len(tt_data)  + 1))

    ax.plot(ne, ncf_data, color=NCF_COLOR, lw=2, marker='o', ms=4,
            label='NCF (val NDCG@10)', zorder=3)
    ax.plot(te, tt_data,  color=TT_COLOR,  lw=2, marker='s', ms=4,
            label='Two-Tower (val NDCG@10)', zorder=3)

    # Best epoch markers
    ax.axvline(ncf_best, color=NCF_COLOR, ls=':', lw=1, alpha=0.6)
    ax.axvline(tt_best,  color=TT_COLOR,  ls=':', lw=1, alpha=0.6)

    # Reference lines
    ax.axhline(ref_val, color=MF_COLOR, ls='--', lw=1.8,
               label=f'{ref_label} (val NDCG@10 = {ref_val:.4f})', zorder=2)
    ax.axhline(pop_val, color=POP_COLOR, ls='--', lw=1.4, alpha=0.8,
               label=f'Popularity (test NDCG@10 = {pop_val:.4f})', zorder=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val NDCG@10', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc='center right', framealpha=0.9)
    ax.grid(True, ls='--', lw=0.5, alpha=0.5)
    ax.set_xlim(left=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


make_plot(
    ncf_ml, tt_ml, ncf_ml_best, tt_ml_best,
    svd_ml_val, 'MF (SVD)', pop_ml_val,
    'Figure 1: Convergence on MovieLens (Explicit Feedback)',
    OUT_DIR / 'convergence_movielens.png',
)

make_plot(
    ncf_am, tt_am, ncf_am_best, tt_am_best,
    als_am_val, 'ALS', pop_am_val,
    'Figure 2: Convergence on Amazon Music (Implicit Feedback)',
    OUT_DIR / 'convergence_amazonmusic.png',
)