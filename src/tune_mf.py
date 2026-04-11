'''
Hyperparameter tuning for Matrix Factorization using the validation set.

Key fix vs original: when generating recommendations for val users, we pass
train_df as the seen-items source (correct), but we also pass val_df rows
that are NOT the candidate being evaluated — i.e. we never leak val-relevant
items into the seen-item filter because they are never in train_df.

Evaluation candidate set: val_df is used exactly as test_df is used in the
main pipeline (same evaluate_model call), so tuning NDCG is directly
comparable to test NDCG.
'''

import argparse
import os
import pandas as pd
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MOVIELENS_CFG, AMAZON_CFG
from models.matrix_factorization import MatrixFactorizationModel
from evaluate import evaluate_model

BASE_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Param grids
# ---------------------------------------------------------------------------
EXPLICIT_GRID = {
    'n_factors': [20, 50],
    'reg':       [0.0001, 0.0005, 0.001],
    'n_epochs':  [100, 200, 300],
}

IMPLICIT_GRID = {
    'n_factors': [50, 100, 200],
    'reg':       [0.001, 0.01, 0.1],
    'n_epochs':  [30, 50, 100],
    'alpha':     [10, 40, 100],
}


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def tune(cfg):
    base     = BASE_DIR / 'data' / 'processed' / cfg.name
    train_df = pd.read_csv(base / 'train.csv')
    val_df   = pd.read_csv(base / 'val.csv')

    print(f'\nDataset : {cfg.name}  ({cfg.feedback_type})')
    print(f'train   : {len(train_df):,} rows,  {train_df["user_idx"].nunique():,} users')
    print(f'val     : {len(val_df):,} rows,  {val_df["user_idx"].nunique():,} users')
    print(f'val relevance rate: {(val_df["relevant"] == 1).mean():.2%}')

    param_grid = EXPLICIT_GRID if cfg.feedback_type == 'explicit' else IMPLICIT_GRID
    keys       = list(param_grid.keys())
    combos     = list(product(*param_grid.values()))
    print(f'\nSearching {len(combos)} hyperparameter combinations...\n')

    # Only evaluate users that appear in both train and val so that
    # the SVD inner-id lookup never hits the ValueError fallback.
    train_users = set(train_df['user_idx'].unique())
    val_users   = val_df['user_idx'].unique().tolist()
    eval_users  = [u for u in val_users if u in train_users]

    if len(eval_users) < len(val_users):
        print(
            f'  Warning: {len(val_users) - len(eval_users)} val users not in train — excluded from eval.\n'
        )

    best_ndcg   = -1.0
    best_params: dict = {}
    all_results: list[dict] = []

    for combo in combos:
        params = dict(zip(keys, combo))
        model  = MatrixFactorizationModel(cfg, **params)
        model.fit(train_df, val_df)

        # Recommend: filter only items seen in train (not val) so we don't
        # accidentally hide val-relevant items from the candidate list.
        recs    = model.recommend(eval_users, train_df, k=10)
        metrics = evaluate_model(recs, val_df, k=10)

        row = {**params, **metrics}
        all_results.append(row)

        ndcg = metrics['NDCG@10']
        flag = '  *** best so far' if ndcg > best_ndcg else ''
        print(f'  {params}  →  NDCG@10: {ndcg:.4f}  HR@10: {metrics["HR@10"]:.4f}{flag}')

        if ndcg > best_ndcg:
            best_ndcg   = ndcg
            best_params = params

    print(f'\nBest params : {best_params}')
    print(f'Best NDCG@10 on val : {best_ndcg:.4f}')

    # Save results
    out_dir = BASE_DIR / 'results' / cfg.name
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / 'mf_tuning.csv'
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f'\nFull tuning results saved to {out_path}')

    return best_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()

    cfg  = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    best = tune(cfg)