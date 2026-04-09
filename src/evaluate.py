'''
Evaluation metrics for recommender systems.
All functions take a recommendations dict {user_idx: [ranked item_idx, ...]}
and a test_df with columns [user_idx, item_idx, relevant].
'''

import numpy as np
import pandas as pd


def _get_relevant_items(test_df: pd.DataFrame) -> dict[int, set[int]]:
    '''Returns {user_idx: set of relevant item_idx} from test set.
    For movielens, relevant == 1 means rating >= relevance_threshold.
    For amazonmusic, all interactions are set to relevant = 1 as we use implicit feedback.
    Filters out item_idx == -1 (unseen items that were never encoded).
    '''
    return (
        test_df[(test_df['relevant'] == 1) & (test_df['item_idx'] != -1)]
        .groupby('user_idx')['item_idx']
        .apply(set)
        .to_dict()
    )


def hit_rate_at_k(recommendations: dict, test_df: pd.DataFrame, k: int = 10) -> float:
    '''
    Fraction of users for whom at least one relevant item appears in top-k.
    HR@K = (number of users with at least one hit) / (total users evaluated)
    Users with no relevant items are included in the denominator (counted as misses).
    '''
    relevant_items = _get_relevant_items(test_df)
    hits = 0

    for user_idx, recommended_items in recommendations.items():
        top_k = set(recommended_items[:k])
        items_relevant_to_user = relevant_items.get(user_idx, set())
        if top_k.intersection(items_relevant_to_user):
            hits += 1

    return hits / len(recommendations) if recommendations else 0.0


def ndcg_at_k(recommendations: dict, test_df: pd.DataFrame, k: int = 10) -> float:
    '''
    Normalized Discounted Cumulative Gain at K.
    Rewards relevant items ranked higher in the list.
    NDCG@K = (1/|users|) * sum_users [ DCG@K / IDCG@K ]
    Users with no relevant items contribute 0 to the mean.
    '''
    relevant_items = _get_relevant_items(test_df)
    ndcg_scores = []

    for user_idx, ranked_items in recommendations.items():
        top_k = ranked_items[:k]
        user_relevant = relevant_items.get(user_idx, set())

        # DCG: sum of 1/log2(rank+1) for each hit
        dcg = sum(
            1.0 / np.log2(rank + 2)   # rank is 0-indexed so +2 = +1 for 1-indexed +1 for log
            for rank, item in enumerate(top_k)
            if item in user_relevant
        )

        # IDCG: best possible DCG given number of relevant items
        n_relevant = min(len(user_relevant), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(n_relevant))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def mean_average_precision(recommendations: dict, test_df: pd.DataFrame) -> float:
    '''
    Mean Average Precision across all users.
    Users with no relevant items are excluded from the mean (same as standard IR convention).
    '''
    relevant_items = _get_relevant_items(test_df)
    ap_scores = []

    for user_idx, ranked_items in recommendations.items():
        user_relevant = relevant_items.get(user_idx, set())
        if not user_relevant:
            continue

        hits = 0
        precision_sum = 0.0

        for rank, item in enumerate(ranked_items, start=1):
            if item in user_relevant:
                hits += 1
                precision_sum += hits / rank

        ap = precision_sum / len(user_relevant)
        ap_scores.append(ap)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def evaluate_model(recommendations: dict, test_df: pd.DataFrame, k: int = 10) -> dict:
    '''
    Runs all three metrics and returns a results dict.
    Filters test_df to only users present in recommendations before evaluating.
    '''
    evaluated_users = set(recommendations.keys())
    test_evaluated = test_df[test_df['user_idx'].isin(evaluated_users)]

    return {
        f'HR@{k}':   hit_rate_at_k(recommendations, test_evaluated, k),
        f'NDCG@{k}': ndcg_at_k(recommendations, test_evaluated, k),
        f'MAP':  mean_average_precision(recommendations, test_evaluated),
    }

def evaluate_all(recommendations: dict, test_df: pd.DataFrame,
                 cold_start_test_df: pd.DataFrame, k: int = 10) -> dict:
    '''
    Returns metrics for regular and cold-start users separately.
    Splits recommendations by user type before evaluation to avoid any cross-contamination.
    '''
    results = {}

    # regular users
    regular_user_idxs = set(test_df['user_idx'].unique())
    regular_recs = {u: v for u, v in recommendations.items() if u in regular_user_idxs}
    results['regular'] = evaluate_model(regular_recs, test_df, k)

    print(f"Regular users evaluated: {len(regular_recs)}")
    print(f"Cold-start interactions: {len(cold_start_test_df)}")
    print(f"Total recommendations generated: {len(recommendations)}")

    # cold-start users — evaluate against test portion only, not context
    if not cold_start_test_df.empty:
        cold_user_idxs = set(cold_start_test_df['user_idx'].unique())
        cold_recs = {u: v for u, v in recommendations.items() if u in cold_user_idxs}
        if cold_recs:
            results['cold_start'] = evaluate_model(cold_recs, cold_start_test_df, k)

    return results