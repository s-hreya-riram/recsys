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

    def __init__(
        self,
        cfg: DatasetConfig,
        n_factors: int = None,
        reg: float = None,
        n_epochs: int = None,
        alpha: float = None,
    ):
        self.cfg = cfg
        if cfg.feedback_type == 'explicit':
            self.n_factors = n_factors if n_factors is not None else 50
            self.reg       = reg       if reg       is not None else 0.001
            self.n_epochs  = n_epochs  if n_epochs  is not None else 300
            self.alpha     = None
        else:
            self.n_factors = n_factors if n_factors is not None else 200
            self.reg       = reg       if reg       is not None else 0.01
            self.n_epochs  = n_epochs  if n_epochs  is not None else 30
            self.alpha     = alpha     if alpha     is not None else 100

        self.model        = None
        self.user_factors = None   # (n_users, n_factors)
        self.item_factors = None   # (n_items, n_factors)
        self.n_items      = None
        self.n_items_train = None  # width of matrix passed to ALS fit — needed for recalculate_user

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        if self.cfg.feedback_type == 'explicit':
            self._fit_svd(train_df)
        else:
            self._fit_als(train_df)

    def _fit_svd(self, train_df: pd.DataFrame) -> None:
        from surprise import SVD, Dataset, Reader

        print(f'    training SVD: factors={self.n_factors}, reg={self.reg}, epochs={self.n_epochs}')

        min_r, max_r = train_df['rating'].min(), train_df['rating'].max()
        reader    = Reader(rating_scale=(min_r, max_r))
        data      = Dataset.load_from_df(train_df[['user_idx', 'item_idx', 'rating']], reader)
        trainset  = data.build_full_trainset()

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            reg_all=self.reg,
            verbose=False,
        )
        self.model.fit(trainset)

        # Extract Surprise's internal factors into dense arrays indexed by
        # YOUR item_idx / user_idx so _recommend_svd never needs to round-trip
        # through Surprise's inner ID mapping (which is the source of ranking bugs).
        n_users = train_df['user_idx'].max() + 1
        n_items = train_df['item_idx'].max() + 1
        self.n_items = n_items

        self.user_factors = np.zeros((n_users, self.n_factors))
        self.item_factors = np.zeros((n_items, self.n_factors))

        for inner_uid in range(trainset.n_users):
            raw_uid = int(trainset.to_raw_uid(inner_uid))
            self.user_factors[raw_uid] = self.model.pu[inner_uid]

        for inner_iid in range(trainset.n_items):
            raw_iid = int(trainset.to_raw_iid(inner_iid))
            self.item_factors[raw_iid] = self.model.qi[inner_iid]

    def _fit_als(self, train_df: pd.DataFrame) -> None:
        import implicit
        from scipy.sparse import csr_matrix

        print(f'    training ALS: factors={self.n_factors}, reg={self.reg}, iterations={self.n_epochs}, alpha={self.alpha}')

        n_users = train_df['user_idx'].max() + 1
        n_items = train_df['item_idx'].max() + 1
        self.n_items       = n_items
        self.n_items_train = n_items  # recalculate_user requires matrix width == training width

        train_agg = train_df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
        user_item = csr_matrix(
            (train_agg['count'].values, (train_agg['user_idx'], train_agg['item_idx'])),
            shape=(n_users, n_items),
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.reg,
            iterations=self.n_epochs,
            calculate_training_loss=True,
        )
        self.model.fit(user_item * self.alpha)

        self.user_factors = self.model.user_factors   # (n_users, n_factors)
        self.item_factors = self.model.item_factors   # (n_items, n_factors)

    # ------------------------------------------------------------------
    # Recommend (regular users)
    # ------------------------------------------------------------------

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        if self.cfg.feedback_type == 'explicit':
            return self._recommend_svd(user_ids, train_df, k)
        else:
            return self._recommend_als(user_ids, train_df, k)

    def _recommend_svd(
        self, user_ids: list, train_df: pd.DataFrame, k: int
    ) -> dict[int, list[int]]:
        # Identical in structure to _recommend_als — no Surprise inner ID
        # lookups needed since _fit_svd already mapped factors to raw indices.
        seen_items = {u: set(g) for u, g in train_df.groupby('user_idx')['item_idx']}

        recommendations: dict[int, list[int]] = {}

        for user_idx in user_ids:
            user_seen = seen_items.get(user_idx, set())

            if user_idx >= self.user_factors.shape[0]:
                recommendations[user_idx] = []
                continue

            user_vec = self.user_factors[user_idx]
            scores   = self.item_factors @ user_vec

            recommendations[user_idx] = [
                int(i) for i in np.argsort(scores)[::-1]
                if int(i) not in user_seen
            ][:k]

        return recommendations

    def _recommend_als(
        self, user_ids: list, train_df: pd.DataFrame, k: int
    ) -> dict[int, list[int]]:
        seen_items = (
            train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        )

        recommendations: dict[int, list[int]] = {}

        for user_idx in user_ids:
            user_seen = seen_items.get(user_idx, set())

            if user_idx >= self.user_factors.shape[0]:
                recommendations[user_idx] = []
                continue

            user_vec = self.user_factors[user_idx]
            scores   = self.item_factors @ user_vec

            recommendations[user_idx] = [
                int(i) for i in np.argsort(scores)[::-1]
                if int(i) not in user_seen
            ][:k]

        return recommendations

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------

    def recommend_cold_start(
        self,
        user_ids: list,
        context_df: pd.DataFrame,
        train_df: pd.DataFrame,
        k: int = 10,
    ) -> dict[int, list[int]]:
        if self.cfg.feedback_type == 'explicit':
            # SVD has no fold-in — fall back to base (popularity)
            return super().recommend_cold_start(user_ids, context_df, train_df, k)
        else:
            return self._recommend_cold_start_als(user_ids, context_df, train_df, k)

    def _recommend_cold_start_als(
        self,
        user_ids: list,
        context_df: pd.DataFrame,
        train_df: pd.DataFrame,
        k: int,
    ) -> dict[int, list[int]]:
        from scipy.sparse import csr_matrix

        # n_items_full: wide enough to hold any context item index without crashing.
        # n_items_train: what recalculate_user expects (must match training matrix width).
        # Context items beyond n_items_train have no learnt factors, so trimming is lossless.
        max_context_item  = context_df['item_idx'].replace(-1, 0).max()
        n_items_full  = max(self.n_items, int(max_context_item) + 1)
        n_items_train = self.n_items_train

        popular_items: list[int] = (
            train_df.groupby('item_idx').size()
            .sort_values(ascending=False).index.tolist()
        )

        if context_df.empty or 'item_idx' not in context_df.columns:
            return {uid: popular_items[:k] for uid in user_ids}

        valid_context = context_df[context_df['item_idx'] != -1]
        user_context  = valid_context.groupby('user_idx')['item_idx'].apply(list).to_dict()

        recommendations: dict[int, list[int]] = {}
        n_fallback  = 0
        n_embedding = 0

        for user_idx in user_ids:
            context_items = user_context.get(user_idx, [])
            user_seen     = set(context_items)

            if not context_items:
                recommendations[user_idx] = [i for i in popular_items if i not in user_seen][:k]
                n_fallback += 1
                continue

            n_embedding += 1
            data = np.ones(len(context_items))
            cols = np.array(context_items)

            # Build full-width matrix first (avoids index overflow on unseen items),
            # then trim to train width before passing to recalculate_user.
            user_interactions = csr_matrix(
                (data, (np.zeros(len(context_items), dtype=int), cols)),
                shape=(1, n_items_full),
            ) * self.alpha
            user_interactions_trimmed = user_interactions[:, :n_items_train]

            user_vec = self.model.recalculate_user(0, user_interactions_trimmed)
            scores   = self.item_factors @ user_vec

            recommendations[user_idx] = [
                int(i) for i in np.argsort(scores)[::-1]
                if int(i) not in user_seen
            ][:k]

        print(
            f'    [{self.name}] cold-start: {n_embedding} embedding, {n_fallback} popularity fallback '
            f'({100 * n_fallback / len(user_ids):.0f}% fallback)'
        )
        return recommendations