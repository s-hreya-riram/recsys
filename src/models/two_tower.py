'''
Two-Tower model for recommendation.
Separate user and item encoder networks combined via dot product.
The simple inner product enables fast approximate nearest neighbour
retrieval at inference time — standard architecture for candidate
generation in industry (YouTube, Google Play, etc.)
Reference: Covington et al. 2016 - Deep Neural Networks for YouTube Recommendations
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
# Dataset — same negative sampling strategy as NCF
# ------------------------------------------------------------------

class TwoTowerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_items: int, n_neg: int = 4):
        self.n_items = n_items
        self.n_neg   = n_neg
        self.user_positives = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        self.users   = df['user_idx'].values
        self.items   = df['item_idx'].values

    def __len__(self):
        return len(self.users) * (1 + self.n_neg)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.n_neg)
        is_neg  = idx  % (1 + self.n_neg) != 0
        user    = self.users[pos_idx]

        if not is_neg:
            return (torch.tensor(user),
                    torch.tensor(self.items[pos_idx]),
                    torch.tensor(1.0))

        seen = self.user_positives.get(user, set())
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in seen:
                return torch.tensor(user), torch.tensor(neg_item), torch.tensor(0.0)


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
            in_dim  = out_dim
        # final layer projects to output dim without activation
        # so dot product space is unconstrained
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

        # dot product then sigmoid
        score = torch.sigmoid((user_vec * item_vec).sum(dim=-1))  # (batch,)
        return score


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
                 n_epochs:     int   = 20,
                 patience:     int   = 3):

        self.cfg          = cfg
        if self.cfg.feedback_type != 'implicit':
            # Best params: {'emb_dim': 32, 'tower_layers': [64], 'lr': 0.001, 'n_neg': 4}
            self.emb_dim      = 32
            self.tower_layers = [64]
            self.lr           = 0.001
            self.n_neg        = 4
        else:
            self.emb_dim      = 32
            self.tower_layers = [64]
            self.lr           = 0.005
            self.n_neg        = 8

        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.patience     = patience

        self.model   = None
        self.n_users = None
        self.n_items = None
        self.device  = torch.device('mps' if torch.backends.mps.is_available()
                                    else 'cuda' if torch.cuda.is_available()
                                    else 'cpu')

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.n_users = train_df['user_idx'].max() + 1
        self.n_items = train_df['item_idx'].max() + 1

        print(f'    two-tower training on {self.device}: emb={self.emb_dim}, '
              f'layers={self.tower_layers}, neg={self.n_neg}, lr={self.lr}')

        self.model = TwoTowerArchitecture(
            self.n_users, self.n_items, self.emb_dim, self.tower_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn   = nn.BCELoss()

        train_dataset = TwoTowerDataset(train_df, self.n_items, self.n_neg)
        val_dataset   = TwoTowerDataset(val_df,   self.n_items, self.n_neg)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=0)

        best_val_loss  = float('inf')
        patience_count = 0
        best_weights   = None

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0
            for users, items, labels in train_loader:
                users, items, labels = (users.to(self.device),
                                        items.to(self.device),
                                        labels.to(self.device))
                optimizer.zero_grad()
                preds = self.model(users, items)
                loss  = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for users, items, labels in val_loader:
                    users, items, labels = (users.to(self.device),
                                            items.to(self.device),
                                            labels.to(self.device))
                    val_loss += loss_fn(self.model(users, items), labels).item()

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

        # precompute all item embeddings once — this is the key efficiency
        # advantage of two-tower over NCF at inference time
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

                # dot product with all items at once
                scores = (all_item_vecs * user_vec).sum(dim=-1).cpu().numpy()

                ranked = [
                    int(i) for i in np.argsort(scores)[::-1]
                    if int(i) not in user_seen
                ][:k]

                recommendations[user_idx] = ranked

        return recommendations