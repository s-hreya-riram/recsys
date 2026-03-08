import pandas as pd
import os

def load_movielens_ratings(path='../../data/raw/movielens/ratings.csv'):
    ratings = pd.read_csv(path)
    ratings = ratings.dropna(subset=['userId', 'movieId', 'timestamp'])
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings

def encode_ids(ratings):
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    ratings["user_idx"] = ratings["userId"].map(user_id_map)
    ratings["movie_idx"] = ratings["movieId"].map(movie_id_map)
    return ratings, user_id_map, movie_id_map

def temporal_split(ratings, train_frac=0.8, val_frac=0.1):
    train_list, val_list, test_list = [], [], []
    
    for user_id, group in ratings.groupby('userId'):
        group = group.sort_values('timestamp')
        n = len(group)
        if n < 3:  # Skip tiny users
            train_list.append(group)
            continue
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])
    
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    
    return train_df, val_df, test_df


def write_splits(train_df, val_df, test_df, output_dir='../../data/processed/movielens'):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    print(f"Saved splits to {output_dir}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    ratings = load_movielens_ratings()
    ratings, user_map, movie_map = encode_ids(ratings)  # Fixed!
    train_df, val_df, test_df = temporal_split(ratings)
    write_splits(train_df, val_df, test_df)
