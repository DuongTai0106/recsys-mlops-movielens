import pandas as pd

def popularity_ranking(train_df: pd.DataFrame) -> pd.Series:
    # Returns Series indexed by item_id with popularity count descending
    pop = train_df["item_id"].value_counts()
    return pop

def recommend_popularity(user_id: int, train_df: pd.DataFrame, pop_rank: pd.Series, k: int = 20):
    seen = set(train_df.loc[train_df["user_id"] == user_id, "item_id"].tolist())
    recs = [int(i) for i in pop_rank.index if int(i) not in seen]
    return recs[:k]
