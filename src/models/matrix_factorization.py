'''
Matrix Factorization model for explicit and implicit feedback.
Uses SVD (via Surprise) for explicit feedback (MovieLens)
and ALS (via implicit library) for implicit feedback (Amazon Music).
'''

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.base import BaseModel
from config import DatasetConfig

class MatrixFactorizationModel(BaseModel):
    name = 'matrix_factorization'

    def __init__(self, cfg: DatasetConfig):
        self.cfg       = cfg
        if self.cfg.feedback_type == 'explicit':
            # Best params: (n_factors=100, reg=0.1, n_epochs=30)
            self.n_factors = 100
            self.reg       = 0.1
            self.n_epochs  = 30
        else:
            # Best params: {'n_factors': 100, 'reg': 0.001, 'n_epochs': 50, 'alpha': 10}
            self.n_factors = 100
            self.reg       = 0.001
            self.n_epochs  = 50
            self.alpha     = 10
        self.model          = None
        self.user_factors   = None  # shape (n_users, n_factors)
        self.item_factors   = None  # shape (n_items, n_factors)
        self.n_items        = None

    # Fit either SVD or ALS depending on feedback type
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        if self.cfg.feedback_type == 'explicit':
            self._fit_svd(train_df)
        else:
            self._fit_als(train_df)
            print(f"User factors (first 5): {self.user_factors[:5]}")
            print(f"Item factors (first 5): {self.item_factors[:5]}")
            print(np.isnan(self.user_factors).any(), np.isnan(self.item_factors).any())

    def _fit_svd(self, train_df: pd.DataFrame) -> None:
        from surprise import SVD, Dataset, Reader
        from surprise import accuracy
        import surprise

        print(f'    training SVD: factors={self.n_factors}, reg={self.reg}, epochs={self.n_epochs}')

        # surprise expects ratings in a specific range
        min_r, max_r = train_df['rating'].min(), train_df['rating'].max()
        reader = Reader(rating_scale=(min_r, max_r))

        # surprise needs (user, item, rating) — use encoded integer ids
        data = Dataset.load_from_df(
            train_df[['user_idx', 'item_idx', 'rating']], reader
        )
        trainset = data.build_full_trainset()

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            reg_all=self.reg,
            verbose=False
        )
        self.model.fit(trainset)

        # extract factor matrices for fast batch inference
        self.user_factors = self.model.pu   # (n_users, n_factors)
        self.item_factors = self.model.qi   # (n_items, n_factors)
        self.n_items = train_df['item_idx'].max() + 1

    def _fit_als(self, train_df: pd.DataFrame) -> None:
        import implicit
        from scipy.sparse import csr_matrix

        print(f'    training ALS: factors={self.n_factors}, reg={self.reg}, iterations={self.n_epochs}')

        self.n_users = train_df['user_idx'].max() + 1
        self.n_items = train_df['item_idx'].max() + 1

        # ALS expects a (users, items) sparse matrix of interaction counts
        # alpha scales the confidence: higher interaction count = more confident
        train_agg = train_df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
        user_item_matrix = csr_matrix(
            (train_agg['count'].values, (train_agg['user_idx'], train_agg['item_idx'])),
            shape=(self.n_users, self.n_items)
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.reg,
            iterations=self.n_epochs,
            calculate_training_loss=True,
        )

        self.model.fit(user_item_matrix * self.alpha)

        self.user_factors = self.model.user_factors  # (n_users, n_factors)
        self.item_factors = self.model.item_factors  # (n_items, n_factors) 

        print(f" Shape of user_factors: {self.user_factors.shape}, item_factors: {self.item_factors.shape}")

        user_vec = self.user_factors[0]
        scores = self.item_factors @ user_vec 
        print(scores.min(), scores.max(), scores.std())
        print(f"user_factors shape: {self.user_factors.shape}")  # should be (n_users, n_factors)
        print(f"item_factors shape: {self.item_factors.shape}")  # should be (n_items, n_factors)
        print(f"Expected: users={self.n_users}, items={self.n_items}")

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        if self.cfg.feedback_type == 'explicit':
            return self._recommend_svd(user_ids, train_df, k)
        else:
            return self._recommend_als(user_ids, train_df, k)

    def _recommend_svd(self, user_ids: list, train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        seen_items = (
            train_df.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        )

        trainset = self.model.trainset
        recommendations = {}
        debug_printed = False  # print once across all users, only when ranked is available

        for user_idx in user_ids:
            user_seen = seen_items.get(user_idx, set())

            try:
                inner_uid = trainset.to_inner_uid(user_idx)
                user_vec  = self.user_factors[inner_uid]
                scores    = self.item_factors @ user_vec
                ranked    = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )
                ranked_external = [
                    int(trainset.to_raw_iid(i)) for i in ranked
                    if int(trainset.to_raw_iid(i)) not in user_seen
                ][:k]

                if not debug_printed:
                    sample_raw  = [trainset.to_raw_iid(i) for i in ranked[:3]]
                    sample_seen = list(user_seen)[:3]
                    print(f"  raw iid types: {[type(x) for x in sample_raw]}")
                    print(f"  raw iid values: {sample_raw}")
                    print(f"  seen types: {[type(x) for x in sample_seen]}")
                    debug_printed = True

            except ValueError:
                # user not in training set — pipeline handles cold-start separately
                ranked_external = []

            recommendations[user_idx] = ranked_external

        return recommendations

    def _recommend_als(self, user_ids: list, train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        from scipy.sparse import csr_matrix

        seen_items = (
            train_df.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        )

        n_users = train_df['user_idx'].max() + 1
        n_items = train_df['item_idx'].max() + 1
        counts  = np.ones(len(train_df))
        user_item_matrix = csr_matrix(
            (counts, (train_df['user_idx'], train_df['item_idx'])),
            shape=(n_users, n_items)
        )

        recommendations = {}
        for user_idx in user_ids:
            user_seen = seen_items.get(user_idx, set())

            if user_idx >= self.user_factors.shape[0]:
                # truly unseen user — no factors, return empty
                recommendations[user_idx] = []
                continue

            user_vec = self.user_factors[user_idx]        # (n_factors,)
            scores   = self.item_factors @ user_vec       # (n_items,)

            ranked = [
                int(i) for i in np.argsort(scores)[::-1]
                if int(i) not in user_seen
            ][:k]

            recommendations[user_idx] = ranked

        return recommendations
    
    def recommend_cold_start(self, user_ids: list, context_df: pd.DataFrame,
                            train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        if self.cfg.feedback_type == 'explicit':
            # SVD has no fold-in support — fall back to popularity
            return super().recommend_cold_start(user_ids, context_df, train_df, k)
        else:
            return self._recommend_cold_start_als(user_ids, context_df, train_df, k)

    def _recommend_cold_start_als(self, user_ids: list, context_df: pd.DataFrame,
                                train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        from scipy.sparse import csr_matrix

        n_fallback = 0
        n_embedding = 0
        n_items = train_df['item_idx'].max() + 1

        # precompute popular items for fallback
        popular_items = (
            train_df.groupby('item_idx').size()
            .sort_values(ascending=False).index.tolist()
        )

        if context_df.empty or 'item_idx' not in context_df.columns:
            # fall back to popularity for all users
            popular_items = (
                train_df.groupby('item_idx').size()
                .sort_values(ascending=False).index.tolist()
            )
            return {uid: popular_items[:k] for uid in user_ids}

        valid_context = context_df[context_df['item_idx'] != -1]

        user_context  = (
            valid_context.groupby('user_idx')['item_idx']
            .apply(list).to_dict()
        )

        recommendations = {}
        for user_idx in user_ids:
            context_items = user_context.get(user_idx, [])
            user_seen     = set(context_items)

            if not context_items:
                # no valid context — fall back to popularity
                recommendations[user_idx] = [
                    i for i in popular_items if i not in user_seen
                ][:k]
                n_fallback += 1
                continue

            n_embedding += 1
            # build sparse interaction vector for this user
            data = np.ones(len(context_items))
            rows = np.zeros(len(context_items), dtype=int)
            cols = np.array(context_items)

            # fold-in: compute optimal user vector given fixed item factors
            # recalculate_user returns the user vector that minimises ALS loss
            # for this user's interactions while keeping all item factors fixed
            user_interactions = csr_matrix(
                (data, (rows, cols)), shape=(1, n_items)
            ) * self.alpha

            user_vec = self.model.recalculate_user(0, user_interactions)
            scores = self.item_factors @ user_vec

            ranked = [
                int(i) for i in np.argsort(scores)[::-1]
                if int(i) not in user_seen
            ][:k]
            recommendations[user_idx] = ranked

        print(f'    [{self.name}] cold-start: {n_embedding} embedding, {n_fallback} popularity fallback '
          f'({100*n_fallback/len(user_ids):.0f}% fallback)')
        return recommendations