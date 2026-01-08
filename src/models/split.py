import pandas as pd

def leave_last_out(interactions: pd.DataFrame):
    """
    interactions columns: user_id, item_id, ts (sorted or not)
    Returns: train_df, test_df where test is last interaction per user.
    """
    df = interactions.sort_values(["user_id", "ts"]).copy()
    last_idx = df.groupby("user_id")["ts"].idxmax()
    test_df = df.loc[last_idx].copy()
    train_df = df.drop(index=last_idx).copy()
    return train_df, test_df
