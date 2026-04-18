'''
Generate result plots for the recommender systems report.

Bar charts  — HR@10, NDCG@10, MAP per dataset (regular + cold-start)
Radar chart — single combined chart with five dimensions:
                explicit NDCG@10, implicit NDCG@10,
                cold-start NDCG@10 (avg ML+AM), train speed, inference speed
              All axes normalised so larger = better.
              For speed axes the fastest model scores 1.0.

Outputs saved to results/figures/:
  regular_bar_movielens.png
  regular_bar_amazonmusic.png
  cold_start_bar.png
  radar_combined.png

Usage:
    python src/plot_results.py

Requirements: matplotlib, numpy
'''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / 'results' / 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Identity ──────────────────────────────────────────────────────────────────

MODELS       = ['popularity', 'matrix_factorization', 'ncf', 'two_tower']
MODEL_LABELS = ['Popularity', 'Matrix Fact.', 'NCF', 'Two-Tower']
COLORS       = ['#73726c', '#185FA5', '#3B6D11', '#993C1D']
METRICS      = ['HR@10', 'NDCG@10', 'MAP']

# ── Data ──────────────────────────────────────────────────────────────────────

MOVIELENS = {
    'popularity':           [0.1828, 0.0366, 0.0136],
    'matrix_factorization': [0.0860, 0.0152, 0.0051],
    'ncf':                  [0.2430, 0.0433, 0.0152],
    'two_tower':            [0.2323, 0.0444, 0.0171],
}
AMAZON_MUSIC = {
    'popularity':           [0.0151, 0.0076, 0.0058],
    'matrix_factorization': [0.0807, 0.0397, 0.0296],
    'ncf':                  [0.0212, 0.0088, 0.0059],
    'two_tower':            [0.0315, 0.0138, 0.0094],
}
COLD_START = {
    'movielens': {
        'popularity':           [0.5310, 0.1343, 0.0480],
        'matrix_factorization': [0.5310, 0.1343, 0.0480],
        'ncf':                  [0.5379, 0.1118, 0.0329],
        'two_tower':            [0.5379, 0.1423, 0.0507],
    },
    'amazonmusic': {
        'popularity':           [0.0312, 0.0224, 0.0195],
        'matrix_factorization': [0.0938, 0.0619, 0.0517],
        'ncf':                  [0.0156, 0.0156, 0.0156],
        'two_tower':            [0.0625, 0.0314, 0.0215],
    },
}
# Train and eval (inference) times in seconds — lower is better.
# Averaged across both datasets for the combined radar.
SPEED = {
    'movielens':   {'train': [0.0015,  2.8637,  210.4869,  311.0242], 'eval': [0.0361,  1.1305,  1.8934,  1.2916]},
    'amazonmusic': {'train': [0.0024, 24.4871, 1488.5948, 1957.6962], 'eval': [0.1198, 45.9636, 79.0003, 42.3255]},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def model_legend_handles():
    return [
        mpatches.Patch(color=COLORS[i], label=MODEL_LABELS[i], alpha=0.88)
        for i in range(len(MODELS))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Bar charts  (HR@10, NDCG@10, MAP)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_grouped_bars(ax, data: dict, y_max_pad: float = 1.28):
    n_metrics = len(METRICS)
    n_models  = len(MODELS)
    bar_w     = 0.16
    group_gap = 0.25
    x_centers = np.arange(n_metrics) * (n_models * bar_w + group_gap)

    all_vals = [v for m in MODELS for v in data[m]]
    y_max    = max(all_vals)

    for mi, model in enumerate(MODELS):
        offsets = x_centers + (mi - n_models / 2 + 0.5) * bar_w
        values  = data[model]
        bars    = ax.bar(
            offsets, values,
            width=bar_w,
            color=COLORS[mi],
            alpha=0.88,
            edgecolor='white',
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.012,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=6.5, color='#374151',
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(METRICS, fontsize=11)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlim(x_centers[0] - 0.45, x_centers[-1] + 0.45)
    ax.set_ylim(0, y_max * y_max_pad)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))


def plot_grouped_bar(data: dict, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    _draw_grouped_bars(ax, data)
    ax.legend(handles=model_legend_handles(), loc='upper right',
              frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_cold_start_bar():
    datasets  = ['movielens', 'amazonmusic']
    ds_labels = ['MovieLens (Explicit)', 'Amazon Music (Implicit)']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle('Figure 3: Cold-Start User Performance (K=10)', fontsize=13,
                 fontweight='bold', y=1.01)

    for ax, ds, ds_label in zip(axes, datasets, ds_labels):
        _draw_grouped_bars(ax, COLD_START[ds])
        ax.set_title(ds_label, fontsize=11, pad=8)

    fig.legend(
        handles=model_legend_handles(),
        loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False, fontsize=10,
    )
    plt.tight_layout()
    path = OUT_DIR / 'cold_start_bar.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Combined radar chart  (5 dimensions, both datasets)
# ══════════════════════════════════════════════════════════════════════════════
#
# Axes (all normalised 0–1, larger = better):
#   0  NDCG@10 explicit        — MOVIELENS[model][1]
#   1  NDCG@10 implicit        — AMAZON_MUSIC[model][1]
#   2  Cold-start NDCG@10      — avg of ML and AM cold-start NDCG@10
#   3  Train speed             — avg train time across datasets (min/value)
#   4  Inference speed         — avg eval time across datasets  (min/value)
#
# Performance axes (0-2): normalised by dividing by max across models.
# Speed axes (3-4): normalised as min_time / model_time so fastest = 1.0.

RADAR_AXIS_LABELS = [
    'NDCG@10\n(explicit)',
    'NDCG@10\n(implicit)',
    'Cold-start\nNDCG@10',
    'Train\nspeed',
    'Inference\nspeed',
]


def _build_combined_radar_values() -> dict:
    raw = {
        model: [
            MOVIELENS[model][1],                                              # explicit NDCG@10
            AMAZON_MUSIC[model][1],                                           # implicit NDCG@10
            (COLD_START['movielens'][model][1]
             + COLD_START['amazonmusic'][model][1]) / 2,                      # avg cold-start NDCG@10
            (SPEED['movielens']['train'][mi]
             + SPEED['amazonmusic']['train'][mi]) / 2,                        # avg train time
            (SPEED['movielens']['eval'][mi]
             + SPEED['amazonmusic']['eval'][mi]) / 2,                         # avg eval time
        ]
        for mi, model in enumerate(MODELS)
    }

    n_axes = len(RADAR_AXIS_LABELS)
    normed = {model: [] for model in MODELS}

    for xi in range(n_axes):
        axis_vals = [raw[m][xi] for m in MODELS]
        if xi < 3:
            mx = max(axis_vals)
            for model in MODELS:
                normed[model].append(raw[model][xi] / mx)
        else:
            mn = min(axis_vals)
            for model in MODELS:
                normed[model].append(mn / raw[model][xi])

    return normed


def plot_combined_radar():
    normed  = _build_combined_radar_values()
    n_axes  = len(RADAR_AXIS_LABELS)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for mi, model in enumerate(MODELS):
        vals = normed[model] + [normed[model][0]]
        ax.plot(angles, vals, color=COLORS[mi], linewidth=2, label=MODEL_LABELS[mi])
        ax.fill(angles, vals, color=COLORS[mi], alpha=0.07)
        ax.scatter(angles[:-1], normed[model], color=COLORS[mi], s=45, zorder=5)

    ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_AXIS_LABELS, fontsize=10)
    ax.set_rlabel_position(38)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=7.5, color='grey')
    ax.set_ylim(0, 1)
    ax.tick_params(pad=8)
    ax.set_title(
        'Figure 6: Model trade-off profile\n'
        '(axes normalised; outer edge = best performance;\n'
        'speed axes averaged across both datasets)',
        fontsize=8, fontweight='bold', pad=24,
    )
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.48, 1.18),
        frameon=False, fontsize=9,
    )

    plt.tight_layout()
    path = OUT_DIR / 'radar_combined.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Saving figures to {OUT_DIR}\n')

    plot_grouped_bar(
        MOVIELENS,
        title='Figure 1: Regular User Performance — MovieLens (Explicit, K=10)',
        filename='regular_bar_movielens.png',
    )
    plot_grouped_bar(
        AMAZON_MUSIC,
        title='Figure 2: Regular User Performance — Amazon Music (Implicit, K=10)',
        filename='regular_bar_amazonmusic.png',
    )
    plot_cold_start_bar()
    plot_combined_radar()