import pandas as pd

class BaseModel:
    name: str

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        raise NotImplementedError

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        '''
        Returns {user_idx: [ranked item_idx, ...]} for top-k unseen items.
        train_df is used to exclude already-seen items.
        '''
        raise NotImplementedError

    def recommend_cold_start(self, user_ids: list, context_df: pd.DataFrame,
                             train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        '''
        Default cold-start strategy: popularity fallback filtered by context interactions.
        
        Certain models that can adopt a different strategy (MF via fold-in, NCF/two-tower via average item 
        embeddings) should override this method.
        
        context_df contains the cold-start users' limited interaction history —
        these items are filtered out of recommendations (user has already seen them).
        train_df is used to compute the global popularity ranking.
        '''
        popular_items = (
            train_df.groupby('item_idx')
            .size()
            .sort_values(ascending=False)
            .index.tolist()
        )
        seen_items = (
            context_df.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        ) if not context_df.empty else {}

        recommendations = {}
        n_fallback = 0  # track fallbacks
        for user_idx in user_ids:
            user_seen = seen_items.get(user_idx, set())
            recommendations[user_idx] = [
                i for i in popular_items if i not in user_seen
            ][:k]
            n_fallback += 1

        print(f'    [{self.name}] cold-start fallback: {n_fallback}/{len(user_ids)} users used popularity')
        return recommendations