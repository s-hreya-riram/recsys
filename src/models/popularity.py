import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.base import BaseModel

class PopularityModel(BaseModel):
    name = "popularity"

    def __init__(self):
        self.popular_items = []

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.popular_items = train_df.groupby('item_idx').size().sort_values(ascending=False).index.tolist()

    def recommend(self, user_ids: list, train_df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        """Returns {user_idx: [ranked movie_idx, ...]} for top-k items.
            Excludes items that the user has already interacted with in the training set.
        """
        recommendations = {}
        # get items user has seen in the training set so as to
        # exclude them from the recommendations
        seen_items_dict = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        for user_id in user_ids:
            results = []

            for ele in self.popular_items:
                if ele not in seen_items_dict.get(user_id, set()):
                    results.append(ele)
                if len(results) == k:
                    break
            recommendations[user_id] = results
        # print(f"Recommendations for user_id 7214: {recommendations.get(7214)}")
        return recommendations



