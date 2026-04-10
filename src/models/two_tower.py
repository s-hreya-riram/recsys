'''
Two-Tower model for recommendation.
Separate user and item encoder networks combined via dot product.
The simple inner product enables fast approximate nearest neighbour
retrieval at inference time. Loss computation is performed using Bayesian Personalised Ranking (BPR), 
this directly optimises ranking
by maximising the score gap between observed (positive) and unobserved
(negative) interactions. More appropriate than pointwise BCE for implicit
feedback where only positive interactions are observed.
Reference: Rendle et al. 2009 - BPR: Bayesian Personalized Ranking from Implicit Feedback
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.base import BaseModel
from config import DatasetConfig


# ------------------------------------------------------------------
# BPR loss
# ------------------------------------------------------------------

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    '''
    Bayesian Personalised Ranking loss.
    Maximises the score margin between positive and negative items.
    Loss = -mean(log(sigmoid(pos_score - neg_score)))
    '''
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


# ------------------------------------------------------------------
# Dataset — returns (user, pos_item, neg_item) triplets for BPR
# ------------------------------------------------------------------

class TwoTowerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_items: int, n_neg: int = 4,
                 global_seen: dict = None):
        '''
        Args:
            df:           interactions dataframe (user_idx, item_idx)
            n_items:      total number of items in the catalogue
            n_neg:        number of negative samples per positive
            global_seen:  {user_idx: set(item_idx)} of ALL seen items across
                          train+val splits — used to avoid sampling true
                          positives as negatives. If None, uses df only.
        '''
        self.n_items = n_items
        self.n_neg   = n_neg
        self.users   = df['user_idx'].values
        self.items   = df['item_idx'].values

        # use global_seen if provided (avoids train items being sampled as
        # negatives when computing val loss), otherwise fall back to df
        if global_seen is not None:
            self.user_positives = global_seen
        else:
            self.user_positives = (
                df.groupby('user_idx')['item_idx'].apply(set).to_dict()
            )

    def __len__(self):
        return len(self.users) * self.n_neg

    def __getitem__(self, idx):
        pos_idx  = idx // self.n_neg
        user     = self.users[pos_idx]
        pos_item = self.items[pos_idx]

        seen = self.user_positives.get(user, set())
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in seen:
                return (torch.tensor(user),
                        torch.tensor(pos_item),
                        torch.tensor(neg_item))


# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------

class TwoTowerArchitecture(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 emb_dim: int, tower_layers: list):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # user tower — MLP on top of user embedding
        self.user_tower = self._build_tower(emb_dim, tower_layers)

        # item tower — same architecture, separate weights
        self.item_tower = self._build_tower(emb_dim, tower_layers)

        self._init_weights()

    def _build_tower(self, emb_dim: int, layers: list) -> nn.Sequential:
        '''Builds MLP: emb_dim -> layers[0] -> ... -> layers[-1]'''
        tower  = []
        in_dim = emb_dim
        for out_dim in layers:
            tower += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = out_dim
        return nn.Sequential(*tower)

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for tower in [self.user_tower, self.item_tower]:
            for layer in tower:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_tower(self.user_emb(user_ids))  # (batch, out_dim)
        item_vec = self.item_tower(self.item_emb(item_ids))  # (batch, out_dim)
        # raw dot product — no sigmoid, so BPR loss operates on unconstrained logits
        return (user_vec * item_vec).sum(dim=-1)             # (batch,)


# ------------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------------

class TwoTowerModel(BaseModel):
    name = 'two_tower'

    def __init__(self, cfg: DatasetConfig,
                 emb_dim:      int   = 32,
                 tower_layers: list  = None,
                 n_neg:        int   = 4,
                 lr:           float = 1e-3,
                 batch_size:   int   = 256,
                 n_epochs:     int   = 10,
                 patience:     int   = 3):

        self.cfg = cfg
        if self.cfg.feedback_type != 'implicit':
            # Best params: {'emb_dim': 64, 'tower_layers': [128, 64], 'lr': 0.001, 'n_neg': 4}
            self.emb_dim      = 64
            self.tower_layers = [128, 64]
            self.lr           = 0.001
            self.n_neg        = 4
        else:
            # Best params: {'emb_dim': 64, 'tower_layers': [128, 64], 'lr': 0.005, 'n_neg': 8}
            self.emb_dim      = 64
            self.tower_layers = [128, 64]
            self.lr           = 0.005
            self.n_neg        = 8

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience   = patience

        self.model   = None
        self.n_users = None
        self.n_items = None
        self.device  = torch.device('mps'  if torch.backends.mps.is_available()  else
                                    'cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        train_df = train_df[
            (train_df['user_idx'] >= 0) & 
            (train_df['item_idx'] >= 0)
        ].copy()
        val_df = val_df[
            (val_df['user_idx'] >= 0) & 
            (val_df['item_idx'] >= 0)
        ].copy()
        self.n_users = train_df['user_idx'].max() + 1
        self.n_items = train_df['item_idx'].max() + 1

        print(f'    two-tower [{self.cfg.feedback_type}] on {self.device}: '
              f'emb={self.emb_dim}, layers={self.tower_layers}, '
              f'neg={self.n_neg}, lr={self.lr}')

        self.model = TwoTowerArchitecture(
            self.n_users, self.n_items, self.emb_dim, self.tower_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # build global seen dict — val negatives must exclude train positives too
        train_seen = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        val_seen   = val_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        global_seen = {
            u: train_seen.get(u, set()) | val_seen.get(u, set())
            for u in set(train_seen) | set(val_seen)
        }

        train_dataset = TwoTowerDataset(train_df, self.n_items, self.n_neg)
        val_dataset   = TwoTowerDataset(val_df,   self.n_items, self.n_neg,
                                        global_seen=global_seen)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=0)

        best_val_loss  = float('inf')
        patience_count = 0
        best_weights   = None

        for epoch in range(self.n_epochs):
            # --- train ---
            self.model.train()
            train_loss = 0.0
            for users, pos_items, neg_items in train_loader:
                users     = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                optimizer.zero_grad()
                pos_scores = self.model(users, pos_items)
                neg_scores = self.model(users, neg_items)
                loss       = bpr_loss(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- val ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for users, pos_items, neg_items in val_loader:
                    users     = users.to(self.device)
                    pos_items = pos_items.to(self.device)
                    neg_items = neg_items.to(self.device)
                    pos_scores = self.model(users, pos_items)
                    neg_scores = self.model(users, neg_items)
                    val_loss  += bpr_loss(pos_scores, neg_scores).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            print(f'      epoch {epoch+1}/{self.n_epochs} '
                  f'train_loss={avg_train:.4f} val_loss={avg_val:.4f}')

            if avg_val < best_val_loss:
                best_val_loss  = avg_val
                patience_count = 0
                best_weights   = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f'      early stopping at epoch {epoch+1}')
                    break

        self.model.load_state_dict(best_weights)
        print(f'      best val loss: {best_val_loss:.4f}')

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        self.model.eval()

        seen_items = (
            train_df.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        )

        # precompute all item embeddings once — key efficiency advantage of
        # two-tower: item tower only runs once regardless of number of users
        all_item_ids = torch.arange(self.n_items).to(self.device)
        with torch.no_grad():
            all_item_vecs = self.model.item_tower(
                self.model.item_emb(all_item_ids)
            )  # (n_items, out_dim)

        recommendations = {}
        with torch.no_grad():
            for user_idx in user_ids:
                if user_idx >= self.n_users:
                    recommendations[user_idx] = []
                    continue

                user_seen = seen_items.get(user_idx, set())
                user_id_t = torch.tensor([user_idx]).to(self.device)
                user_vec  = self.model.user_tower(
                    self.model.user_emb(user_id_t)
                )  # (1, out_dim)

                # dot product with all items — no sigmoid needed at inference,
                # ranking order is preserved with raw scores
                scores = (all_item_vecs * user_vec).sum(dim=-1).cpu().numpy()

                ranked = [
                    int(i) for i in np.argsort(scores)[::-1]
                    if int(i) not in user_seen
                ][:k]

                recommendations[user_idx] = ranked

        return recommendations
    
    def recommend_cold_start(self, user_ids: list, context_df: pd.DataFrame,
                         train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        self.model.eval()

        popular_items = (
            train_df.groupby('item_idx').size()
            .sort_values(ascending=False).index.tolist()
        )

        valid_context = context_df[context_df['item_idx'] != -1]
        user_context  = (
            valid_context.groupby('user_idx')['item_idx']
            .apply(list).to_dict()
        )

        # precompute all item vectors once
        all_item_ids = torch.arange(self.n_items).to(self.device)
        with torch.no_grad():
            all_item_vecs = self.model.item_tower(
                self.model.item_emb(all_item_ids)
            )  # (n_items, out_dim)

        recommendations = {}
        n_fallback = 0
        n_embedding = 0
        with torch.no_grad():
            for user_idx in user_ids:
                context_items = user_context.get(user_idx, [])
                user_seen     = set(context_items)


                if not context_items:
                    n_fallback += 1
                    recommendations[user_idx] = [
                        i for i in popular_items if i not in user_seen
                    ][:k]
                    continue
                n_embedding += 1
                context_tensor = torch.tensor(context_items).to(self.device)

                # average item tower outputs as proxy user vector
                # this uses the full item tower (embedding + MLP), not just embeddings
                # giving a richer representation than raw embeddings alone
                item_vecs  = self.model.item_tower(
                    self.model.item_emb(context_tensor)
                )                                           # (n_context, out_dim)
                user_proxy = item_vecs.mean(dim=0, keepdim=True)  # (1, out_dim)

                scores = (all_item_vecs * user_proxy).sum(dim=-1).cpu().numpy()

                ranked = [
                    int(i) for i in np.argsort(scores)[::-1]
                    if int(i) not in user_seen
                ][:k]
                recommendations[user_idx] = ranked
            print(f'    [{self.name}] cold-start: {n_embedding} embedding, {n_fallback} popularity fallback '
                      f'({100*n_fallback/len(user_ids):.0f}% fallback)')

        return recommendations