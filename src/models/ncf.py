'''
Neural Collaborative Filtering (NCF) model.
Combines Generalized Matrix Factorization (GMF) and MLP branches.
Uses binary cross-entropy with negative sampling for both explicit and implicit feedback.
Reference: He et al. 2017 - Neural Collaborative Filtering
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
# Dataset
# ------------------------------------------------------------------

class InteractionDataset(Dataset):
    '''
    Yields (user_idx, item_idx, label) tuples.
    For each positive interaction, samples n_neg negative items uniformly
    from items not seen by the user.
    '''
    def __init__(self, df: pd.DataFrame, n_items: int, n_neg: int = 4):
        self.n_items = n_items
        self.n_neg   = n_neg

        # group positive items per user for fast negative sampling
        self.user_positives = df.groupby('user_idx')['item_idx'].apply(set).to_dict()

        # store positive pairs
        self.users = df['user_idx'].values
        self.items = df['item_idx'].values

    def __len__(self):
        return len(self.users) * (1 + self.n_neg)

    def __getitem__(self, idx):
        pos_idx  = idx // (1 + self.n_neg)
        is_neg   = idx  % (1 + self.n_neg) != 0

        user = self.users[pos_idx]
        if not is_neg:
            return torch.tensor(user), torch.tensor(self.items[pos_idx]), torch.tensor(1.0)

        # sample a negative item not seen by this user
        seen = self.user_positives.get(user, set())
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in seen:
                return torch.tensor(user), torch.tensor(neg_item), torch.tensor(0.0)


# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------

class NCFArchitecture(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, mlp_layers: list):
        super().__init__()

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP embeddings — separate from GMF
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP layers
        # input size is 2*emb_dim because we concatenate user and item embeddings
        mlp_input_dim = emb_dim * 2
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim  = out_dim
        self.mlp = nn.Sequential(*layers)

        # final prediction layer
        # input: GMF output (emb_dim) + MLP output (last layer size)
        self.predict = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids, item_ids):
        # GMF branch — element-wise product
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_out  = gmf_user * gmf_item                          # (batch, emb_dim)

        # MLP branch — concatenate then pass through layers
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_in   = torch.cat([mlp_user, mlp_item], dim=-1)     # (batch, 2*emb_dim)
        mlp_out  = self.mlp(mlp_in)                            # (batch, last_layer_dim)

        # concatenate GMF and MLP outputs, predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)       # (batch, emb_dim + last_layer_dim)
        score    = torch.sigmoid(self.predict(combined))        # (batch, 1)
        return score.squeeze(-1)


# ------------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------------

class NCFModel(BaseModel):
    name = 'ncf'

    def __init__(self, cfg, emb_dim=32, mlp_layers=None, n_neg=4,
             lr=1e-3, batch_size=256, n_epochs=50, patience=5):
        self.cfg        = cfg

        if self.cfg.feedback_type == 'explicit':
            # Best params: (emb_dim=32, mlp_layers=[128,64,32], lr=0.001, n_neg=8) → NDCG@10=0.0708 ✓
            self.emb_dim    = 32
            self.mlp_layers = mlp_layers or [128, 64, 32]
            self.n_neg      = 8
            self.lr         = 1e-3
        else:
            # use input params for implicit feedback since we don't have a strong baseline for it
            # Best params found in tuning: {64,[128, 64, 32],0.001,4}
            self.emb_dim    = emb_dim or 64
            self.mlp_layers = mlp_layers or [128, 64, 32]
            self.n_neg      = n_neg or 4
            self.lr         = lr or 0.001

        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.patience   = patience

        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.patience   = patience   # early stopping patience (in epochs, based on val NDCG@10)

        self.model    = None
        self.n_users  = None
        self.n_items  = None
        self.device   = torch.device('mps' if torch.backends.mps.is_available()
                                     else 'cuda' if torch.cuda.is_available()
                                     else 'cpu')

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        from evaluate import evaluate_model

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

        print(f'    NCF training on {self.device}: emb={self.emb_dim}, '
              f'layers={self.mlp_layers}, neg={self.n_neg}, lr={self.lr}')

        self.model = NCFArchitecture(
            self.n_users, self.n_items, self.emb_dim, self.mlp_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn   = nn.BCELoss()

        import random

        def seed_worker(worker_id):
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)

        train_dataset = InteractionDataset(train_df, self.n_items, self.n_neg)
        train_loader  = DataLoader(train_dataset, batch_size=self.batch_size,
                                   shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker)

        val_user_ids  = val_df['user_idx'].unique().tolist()

        best_val_ndcg  = -1.0
        patience_count = 0
        best_weights   = None

        for epoch in range(self.n_epochs):
            # --- train ---
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

            avg_train = train_loss / len(train_loader)

            # --- val: evaluate ranking quality directly ---
            # BCE val loss is a poor early stopping signal for ranking tasks —
            # it increases monotonically as the model assigns higher confidence
            # to negatives it has never seen. NDCG@10 on val reflects actual
            # recommendation quality and is the correct signal to use here.
            self.model.eval()
            val_recs    = self.recommend(val_user_ids, train_df, k=10)
            val_metrics = evaluate_model(val_recs, val_df, k=10)
            val_ndcg    = val_metrics['NDCG@10']

            print(f'      epoch {epoch+1}/{self.n_epochs} '
                  f'train_loss={avg_train:.4f} val_NDCG@10={val_ndcg:.4f}')

            # early stopping on val NDCG@10
            if val_ndcg > best_val_ndcg:
                best_val_ndcg  = val_ndcg
                patience_count = 0
                best_weights   = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f'      early stopping at epoch {epoch+1}')
                    break

        self.model.load_state_dict(best_weights)
        print(f'      best val NDCG@10: {best_val_ndcg:.4f}')

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        self.model.eval()

        seen_items = (
            train_df.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        )

        all_items  = torch.arange(self.n_items).to(self.device)
        recommendations = {}

        with torch.no_grad():
            for user_idx in user_ids:
                # skip truly unseen users — no embedding exists for them
                if user_idx >= self.n_users:
                    recommendations[user_idx] = []
                    continue

                user_seen   = seen_items.get(user_idx, set())
                user_tensor = torch.tensor([user_idx] * self.n_items).to(self.device)

                scores = self.model(user_tensor, all_items).cpu().numpy()

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

        all_items  = torch.arange(self.n_items).to(self.device)
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

                # average item embeddings from context as proxy user vector
                # use both GMF and MLP item embeddings since NCF has two branches
                gmf_item_vecs = self.model.gmf_item_emb(context_tensor)
                mlp_item_vecs = self.model.mlp_item_emb(context_tensor)
                proxy_gmf     = gmf_item_vecs.mean(dim=0, keepdim=True)  # (1, emb_dim)
                proxy_mlp     = mlp_item_vecs.mean(dim=0, keepdim=True)  # (1, emb_dim)

                # score all items using proxy vectors
                all_gmf_items = self.model.gmf_item_emb(all_items)       # (n_items, emb_dim)
                all_mlp_items = self.model.mlp_item_emb(all_items)       # (n_items, emb_dim)

                gmf_out   = proxy_gmf * all_gmf_items                    # (n_items, emb_dim)
                mlp_in    = torch.cat([
                    proxy_mlp.expand(self.n_items, -1),
                    all_mlp_items
                ], dim=-1)                                                # (n_items, 2*emb_dim)
                mlp_out   = self.model.mlp(mlp_in)                       # (n_items, last_dim)
                combined  = torch.cat([gmf_out, mlp_out], dim=-1)
                scores    = torch.sigmoid(
                    self.model.predict(combined)
                ).squeeze(-1).cpu().numpy()                               # (n_items,)

                ranked = [
                    int(i) for i in np.argsort(scores)[::-1]
                    if int(i) not in user_seen
                ][:k]
                recommendations[user_idx] = ranked

        print(f'    [{self.name}] cold-start: {n_embedding} embedding, {n_fallback} popularity fallback '
              f'({100*n_fallback/len(user_ids):.0f}% fallback)')

        return recommendations