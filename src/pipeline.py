'''
Main pipeline entry point. Runs preprocessing, training, and evaluation for a given dataset.
'''

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MOVIELENS_CFG, AMAZON_CFG
from data.preprocessing import LOADERS, OUTPUT_DIRS, temporal_split, encode_ids, write_splits, split_cold_start
from models.popularity import PopularityModel
from models.matrix_factorization import MatrixFactorizationModel
from models.ncf import NCFModel
from models.two_tower import TwoTowerModel
from evaluate import evaluate_all

BASE_DIR = Path(__file__).parent.parent

def load_splits(cfg):
    base = BASE_DIR / 'data' / 'processed' / cfg.name
    train_df      = pd.read_csv(base / 'train.csv')
    val_df        = pd.read_csv(base / 'val.csv')
    test_df       = pd.read_csv(base / 'test.csv')
    cold_start_df = pd.read_csv(base / 'cold_start_users.csv')
    return train_df, val_df, test_df, cold_start_df


def write_results(all_results: list, cfg):
    import os
    out_dir = BASE_DIR / 'results' / cfg.name
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'metrics.csv', index=False)
    print(f'\nResults saved to results/{cfg.name}/metrics.csv')
    print(df.to_string(index=False))


def set_seeds(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_pipeline(cfg, skip_preprocess=False):
    set_seeds()
    print(f"\n{'='*50}")
    print(f"Dataset: {cfg.name} ({cfg.feedback_type})")
    print(f"{'='*50}")

    # Preprocessing
    if not skip_preprocess:
        print('\n Executing step 1: preprocessing...')
        ratings = LOADERS[cfg.name]()
        train_df, val_df, test_df, cold_start_df = temporal_split(
            ratings, cold_start_threshold=cfg.cold_start_threshold
        )
        train_df, val_df, test_df, cold_start_df, user_map, item_map = encode_ids(
            train_df, val_df, test_df, cold_start_df
        )
        
        write_splits(train_df, val_df, test_df, cold_start_df,
                     output_dir=OUTPUT_DIRS[cfg.name])
    else:
        print('\n Skipping preprocessing (using existing splits)...')

    # Load splits
    train_df, val_df, test_df, cold_start_df = load_splits(cfg)

    cold_start_context_df, cold_start_test_df = split_cold_start(cold_start_df, context_size=3)

    cold_start_user_ids = (
        cold_start_context_df['user_idx'].unique().tolist()
        if not cold_start_context_df.empty else []
    )

    # Separate user ID lists up front — no mixing between evaluation sets
    regular_user_ids    = test_df['user_idx'].unique().tolist()
    print(f"Sample regular_user_ids: {regular_user_ids[:5]}")
    print(f"Sample test_df user_idx: {test_df['user_idx'].unique()[:5]}")
    print(f"Do they overlap: {set(regular_user_ids[:5]) & set(test_df['user_idx'].unique()[:5])}")

    print(f"\n Unique regular user IDs in test set:  {len(regular_user_ids)}")
    print(f" Unique cold-start user IDs:           {len(cold_start_user_ids)}")

    # Training & Evaluation
    print('\n Executing step 2: training and evaluating models')

    models = [
        PopularityModel(),
        MatrixFactorizationModel(cfg),
        NCFModel(cfg),
        TwoTowerModel(cfg),
    ]

    all_results = []

    import time

    for model in models:
        print(f'\n  Fitting {model.name}...')

        train_start = time.time()
        model.fit(train_df, val_df)
        train_time  = time.time() - train_start
        print(f'  training time: {train_time:.4f}s')

        eval_start     = time.time()
        regular_recs   = model.recommend(regular_user_ids, train_df, k=10)
        cold_start_recs = model.recommend_cold_start(
            cold_start_user_ids, cold_start_context_df, train_df, k=10
        ) if cold_start_user_ids else {}
        eval_time = time.time() - eval_start
        print(f'  eval time: {eval_time:.4f}s')

        recommendations = {**regular_recs, **cold_start_recs}
        results = evaluate_all(recommendations, test_df, cold_start_test_df, k=10)

        row = {
            'model':        model.name,
            'train_time_s': round(train_time, 4),
            'eval_time_s':  round(eval_time, 4),
        }
        for split, metrics in results.items():
            for metric, value in metrics.items():
                row[f'{split}_{metric}'] = round(value, 4)
        all_results.append(row)

        for split, metrics in results.items():
            print(f'    {split}: { {k: round(v, 4) for k, v in metrics.items()} }')


    write_results(all_results, cfg)
    print('\n Completed execution of the pipeline for the chosen dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()

    cfg = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    run_pipeline(cfg)