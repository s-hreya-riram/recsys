'''
Hyperparameter tuning for Matrix Factorization using val set.
'''

import pandas as pd
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import MOVIELENS_CFG, AMAZON_CFG
from models.matrix_factorization import MatrixFactorizationModel
from evaluate import evaluate_model

BASE_DIR = Path(__file__).parent.parent

def tune(cfg):
    base = BASE_DIR / 'data' / 'processed' / cfg.name
    train_df = pd.read_csv(base / 'train.csv')
    val_df   = pd.read_csv(base / 'val.csv')

    # tuning grid — keep it manageable
    if cfg.feedback_type == 'explicit':
        param_grid = {
            'n_factors': [20, 50, 100],
            'reg':       [0.001, 0.01, 0.1],
            'n_epochs':  [20, 50],
        }
    else:
        param_grid = {
            'n_factors': [50, 100],
            'reg':       [0.01, 0.1],
            'n_epochs':  [15, 30],
            'alpha':     [10, 40, 100],
        }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    best_ndcg   = -1
    best_params = None
    all_results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model  = MatrixFactorizationModel(cfg, **params)
        model.fit(train_df, val_df)

        # evaluate on val set — use val as both recommendations target and ground truth
        user_ids = val_df['user_idx'].unique().tolist()
        recs     = model.recommend(user_ids, train_df, k=10)
        metrics  = evaluate_model(recs, val_df, k=10)

        row = {**params, **metrics}
        all_results.append(row)

        print(f"  {params} → NDCG@10: {metrics['NDCG@10']:.4f} HR@10: {metrics['HR@10']:.4f}")

        if metrics['NDCG@10'] > best_ndcg:
            best_ndcg   = metrics['NDCG@10']
            best_params = params

    print(f"\nBest params: {best_params}")
    print(f"Best NDCG@10 on val: {best_ndcg:.4f}")

    # save tuning results
    results_df = pd.DataFrame(all_results)
    out = BASE_DIR / 'results' / cfg.name / 'mf_tuning.csv'
    results_df.to_csv(out, index=False)
    print(f"Full tuning results saved to {out}")

    return best_params

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()
    cfg = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    best = tune(cfg)