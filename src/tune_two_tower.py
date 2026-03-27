'''
Hyperparameter tuning for Two-Tower model using val set.
'''

import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import MOVIELENS_CFG, AMAZON_CFG
from models.two_tower import TwoTowerModel
from evaluate import evaluate_model

BASE_DIR = Path(__file__).parent.parent

def tune(cfg):
    base     = BASE_DIR / 'data' / 'processed' / cfg.name
    train_df = pd.read_csv(base / 'train.csv')
    val_df   = pd.read_csv(base / 'val.csv')

    if cfg.feedback_type == 'explicit':
        param_grid = {
            'emb_dim':      [32, 64],
            'tower_layers': [[64], [64, 32], [128, 64]],
            'lr':           [1e-4, 5e-4, 1e-3],
            'n_neg':        [4, 8],
        }
    else:
        param_grid = {
            'emb_dim':      [32, 64],
            'tower_layers': [[64], [128, 64]],
            'lr':           [1e-4, 5e-4, 1e-3],
            'n_neg':        [8, 32, 64],
        }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    best_ndcg   = -1
    best_params = None
    all_results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model  = TwoTowerModel(
            cfg,
            emb_dim=params['emb_dim'],
            tower_layers=params['tower_layers'],
            lr=params['lr'],
            n_neg=params['n_neg'],
            n_epochs=20,
            patience=5,
        )
        model.fit(train_df, val_df)

        user_ids = val_df['user_idx'].unique().tolist()
        recs     = model.recommend(user_ids, train_df, k=10)
        metrics  = evaluate_model(recs, val_df, k=10)

        row = {
            'emb_dim':      params['emb_dim'],
            'tower_layers': str(params['tower_layers']),
            'lr':           params['lr'],
            'n_neg':        params['n_neg'],
            **metrics
        }
        all_results.append(row)

        print(f"  emb={params['emb_dim']} layers={params['tower_layers']} "
              f"lr={params['lr']} neg={params['n_neg']} "
              f"→ NDCG@10: {metrics['NDCG@10']:.4f} HR@10: {metrics['HR@10']:.4f}")

        if metrics['NDCG@10'] > best_ndcg:
            best_ndcg   = metrics['NDCG@10']
            best_params = params

    print(f"\nBest params: {best_params}")
    print(f"Best NDCG@10 on val: {best_ndcg:.4f}")

    out = BASE_DIR / 'results' / cfg.name / 'two_tower_tuning.csv'
    pd.DataFrame(all_results).to_csv(out, index=False)
    print(f"Full tuning results saved to {out}")

    return best_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()
    cfg    = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    best   = tune(cfg)