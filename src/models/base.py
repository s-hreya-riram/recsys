import pandas as pd
class BaseModel:
    name: str

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        raise NotImplementedError

    def recommend(self, train_df: pd.DataFrame, user_ids: list, k: int) -> dict[int, list[int]]:
        """Returns {user_idx: [ranked movie_idx, ...]} for top-k items.
            Excludes items that the user has already interacted with in the training set.
        """
        raise NotImplementedError