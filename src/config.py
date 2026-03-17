from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str # name of the dataset - movielens/amazonmusic
    feedback_type: str # explicit or implicit
    relevance_threshold: float | None # relevance threshold is needed for explicit, not for implicit
    cold_start_threshold: int # threshold for users with fewer interactions
    rating_col: str | None  # rating column is needed for explicit, not for implicit

MOVIELENS_CFG = DatasetConfig(
    name='movielens',
    feedback_type='explicit',
    relevance_threshold=4.0, # configuring this as 4.0 for now, but this may need to be tuned based on the distribution of ratings in the dataset
    cold_start_threshold=35, # starting value of 35 for now although this may need to be tuned
    rating_col='rating',
)

AMAZON_CFG = DatasetConfig(
    name='amazonmusic',
    feedback_type='implicit',
    relevance_threshold=None, # not needed for implicit feedback as all interactions are of relevance
    cold_start_threshold=5, # starting value of 5 for now although this may need to be tuned
    rating_col=None, # the rating column "overall" will not be used as this is implicit feedback
)