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

    def __init__(self, cfg: DatasetConfig, n_factors=50, reg=0.001, n_epochs=50, alpha=40):
        self.cfg       = cfg
        if self.cfg.feedback_type == 'explicit':
            # Best params: {'n_factors': 50, 'reg': 0.001, 'n_epochs': 50}
            self.n_factors = 50
            self.reg       = 0.001
            self.n_epochs  = 50
        else:
            # Best params: {'n_factors': 100, 'reg': 0.1, 'n_epochs': 15, 'alpha': 10}
            self.n_factors = 100
            self.reg       = 0.1
            self.n_epochs  = 15
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
        counts = np.ones(len(train_df))  # binary — each interaction counts as 1
        user_item_matrix = csr_matrix(
            (counts, (train_df['user_idx'], train_df['item_idx'])),
            shape=(self.n_users, self.n_items)
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.reg,
            iterations=self.n_epochs,
            calculate_training_loss=True,
        )
        # implicit expects (items, users) — transpose
        self.model.fit(user_item_matrix.T * self.alpha)

        self.user_factors = self.model.user_factors  # (n_users, n_factors)
        self.item_factors = self.model.item_factors  # (n_items, n_factors)

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