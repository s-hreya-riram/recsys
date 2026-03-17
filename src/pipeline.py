'''
Main pipeline entry point. Runs preprocessing, training, and evaluation for a given dataset.
'''

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MOVIELENS_CFG, AMAZON_CFG
from data.preprocessing import LOADERS, OUTPUT_DIRS, temporal_split, encode_ids, write_splits
from models.popularity import PopularityModel
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


def run_pipeline(cfg, skip_preprocess=False):
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
        train_df, val_df, test_df, cold_start_df, user_map, item_map = encode_ids(train_df, val_df, test_df, cold_start_df)
        write_splits(train_df, val_df, test_df, cold_start_df,
                     output_dir=OUTPUT_DIRS[cfg.name])
    else:
        print('\n Skipping preprocessing (using existing splits)...')

    # Load splits
    train_df, val_df, test_df, cold_start_df = load_splits(cfg)

    # Training & Evaluation
    print('\n Executing step 2: training and evaluating models')

    models = [
        PopularityModel(),
        # TODO add other models once done
    ]

    all_results = []
    for model in models:
        print(f'\n Fitting the {model.name} model')
        model.fit(train_df, val_df)
        user_ids = (test_df['user_idx'].unique().tolist() + cold_start_df['user_idx'].unique().tolist())
        print(f"Unique regular user IDs in test set: {test_df['user_idx'].nunique()}")
        print(f"Unique cold start user IDs: {cold_start_df['user_idx'].nunique()}")
        recommendations = model.recommend(user_ids, train_df, k=10)
        results = evaluate_all(recommendations, test_df, cold_start_df, k=10)

        # flatten results into one row per model
        row = {'model': model.name}
        for split, metrics in results.items():
            for metric, value in metrics.items():
                row[f'{split}_{metric}'] = round(value, 4)
        all_results.append(row)
        
        # print immediately so you see progress
        for split, metrics in results.items():
            print(f'    {split}: { {k: round(v,4) for k,v in metrics.items()} }')

    # write results post training, generating recommendations and evaluation
    write_results(all_results, cfg)
    print('\n Completed execution of the pipeline for the chosen dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()

    cfg = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    run_pipeline(cfg)