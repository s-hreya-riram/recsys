'''
Preprocess raw rating data for MovieLens and Amazon Music datasets, performing temporal splits and encoding IDs.
'''

import pandas as pd
import os
import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MOVIELENS_CFG, AMAZON_CFG

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

def load_movielens_ratings(path=None):
    path = path or BASE_DIR / 'data' / 'raw' / 'movielens' / 'ratings.csv'
    ratings = pd.read_csv(path)
    ratings = ratings.dropna()
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['relevant'] = (ratings[MOVIELENS_CFG.rating_col] >= MOVIELENS_CFG.relevance_threshold).astype(int)
    ratings = ratings.rename(columns={'movieId': 'itemId'})
    return ratings

def load_amazon_ratings(path=None):
    import json
    path = path or BASE_DIR / 'data' / 'raw' / 'amazonmusic' / 'ratings.json'
    records = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            records.append({
                'userId':    d['reviewerID'],
                'itemId':    d['asin'],
                'timestamp': pd.to_datetime(d['unixReviewTime'], unit='s'),
                'relevant':  1
            })
    df = pd.DataFrame(records).dropna()
    user_counts = df['userId'].value_counts()
    item_counts = df['itemId'].value_counts()
    df = df[df['userId'].isin(user_counts[user_counts >= 3].index)]
    df = df[df['itemId'].isin(item_counts[item_counts >= 3].index)]
    return df

LOADERS = {
    'movielens': load_movielens_ratings,
    'amazonmusic':    load_amazon_ratings,
}

OUTPUT_DIRS = {
    'movielens':   str(BASE_DIR / 'data' / 'processed' / 'movielens'),
    'amazonmusic': str(BASE_DIR / 'data' / 'processed' / 'amazonmusic'),
}

def temporal_split(ratings, train_frac=0.8, val_frac=0.1, cold_start_threshold=10):
    train_list, val_list, test_list = [], [], []
    cold_start_list = []  # their interactions go here, not in train

    # sorting group values by timestamp and then performing the split
    for _, group in ratings.groupby('userId'):
        group = group.sort_values('timestamp')
        n = len(group)

        # Cold-start users are held out entirely from training
        # This is so as to build separate models for regular and coldstart users
        if n < cold_start_threshold:
            cold_start_list.append(group)
            continue

        # Ensuring that the train/val/split allows for atleast 1 entry for a group 
        # to be in val and test respectively 
        train_end = min(int(n * train_frac), n - 2)
        val_end   = min(int(n * (train_frac + val_frac)), n - 1)

        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])

    train_df      = pd.concat(train_list,      ignore_index=True)
    val_df        = pd.concat(val_list,        ignore_index=True)
    test_df       = pd.concat(test_list,       ignore_index=True)
    cold_start_df = pd.concat(cold_start_list, ignore_index=True) if cold_start_list else pd.DataFrame()

    return train_df, val_df, test_df, cold_start_df

def encode_ids(train_df, val_df, test_df, cold_start_df):
    """
    Fit ID maps on train only, then apply to val/test.
    Unknown IDs in val/test get -1 (truly unseen).
    """
    user_id_map  = {uid: idx for idx, uid in enumerate(train_df['userId'].unique())}
    item_id_map = {mid: idx for idx, mid in enumerate(train_df['itemId'].unique())}

    for df in [train_df, val_df, test_df, cold_start_df]:
        df['user_idx']  = df['userId'].map(user_id_map).fillna(-1).astype(int)
        df['item_idx'] = df['itemId'].map(item_id_map).fillna(-1).astype(int)

    # assign cold-start users unique indices starting after train users
    if not cold_start_df.empty:
        next_idx = len(user_id_map)
        cold_user_map = {
            uid: next_idx + i 
            for i, uid in enumerate(cold_start_df['userId'].unique())
        }
        cold_start_df['user_idx'] = cold_start_df['userId'].map(cold_user_map).astype(int)
        # items may or may not be in train — use -1 for unseen items
        cold_start_df['item_idx'] = cold_start_df['itemId'].map(item_id_map).fillna(-1).astype(int)

    return train_df, val_df, test_df, cold_start_df, user_id_map, item_id_map

def write_splits(train_df, val_df, test_df, cold_start_df,
                 output_dir='../../data/processed/movielens'):
    os.makedirs(output_dir, exist_ok=True)
    # shuffle the train_df before writing, using a seed to ensure reproducibility
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv',     index=False)
    test_df.to_csv(f'{output_dir}/test.csv',   index=False)

    # Save cold-start user IDs for separate evaluation later
    cold_start_df.to_csv(f'{output_dir}/cold_start_users.csv', index=False)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    cold_start_user_count = cold_start_df['userId'].nunique() if not cold_start_df.empty else 0
    print(f"Cold-start users: {cold_start_user_count:,}")

def split_cold_start(cold_start_df: pd.DataFrame, context_size: int = 3):
    '''
    Splits cold-start user interactions into context and test portions.
    
    Context = first context_size interactions (chronological) — used as 
    input to the model at inference time to approximate user preference.
    
    Test = remaining interactions — used as ground truth for evaluation.
    
    Users with <= context_size interactions are skipped entirely since
    there is nothing left to evaluate against after context is used.
    '''
    context_list = []
    test_list    = []
    skipped      = 0

    for _, group in cold_start_df.groupby('userId'):
        group = group.sort_values('timestamp')
        n     = len(group)

        if n <= context_size:
            # not enough interactions to have both context and test
            skipped += 1
            continue

        context_list.append(group.iloc[:context_size])
        test_list.append(group.iloc[context_size:])

    context_df = pd.concat(context_list, ignore_index=True) if context_list else pd.DataFrame()
    test_df    = pd.concat(test_list,    ignore_index=True) if test_list    else pd.DataFrame()

    print(f"Cold-start split: {context_df['userId'].nunique() if not context_df.empty else 0} users with context, "
          f"{skipped} skipped (too few interactions)")

    return context_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['movielens', 'amazonmusic'], required=True)
    args = parser.parse_args()

    cfg = MOVIELENS_CFG if args.dataset == 'movielens' else AMAZON_CFG
    
    ratings  = LOADERS[args.dataset]()
    train_df, val_df, test_df, cold_start_df = temporal_split(ratings, cold_start_threshold=cfg.cold_start_threshold)
    train_df, val_df, test_df, cold_start_df, user_map, item_map = encode_ids(train_df, val_df, test_df, cold_start_df)
    write_splits(train_df, val_df, test_df, cold_start_df, output_dir=OUTPUT_DIRS[args.dataset])
