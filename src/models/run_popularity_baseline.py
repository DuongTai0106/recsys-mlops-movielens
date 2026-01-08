import pandas as pd

from src.common.config import DATA_PROCESSED_DIR
from src.models.split import leave_last_out
from src.models.baselines import popularity_ranking, recommend_popularity
from src.models.eval import precision_at_k, recall_at_k, ndcg_at_k

def main():
    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")

    train_df, test_df = leave_last_out(interactions)

    pop_rank = popularity_ranking(train_df)

    ks = [10, 20, 50]
    metrics = {k: {"p": [], "r": [], "n": []} for k in ks}

    # truth: last item per user (from test)
    truth_map = dict(zip(test_df["user_id"].astype(int), test_df["item_id"].astype(int)))

    users = list(truth_map.keys())[:5000]  # limit for speed; you can increase later
    for u in users:
        truth = {truth_map[u]}
        recs = recommend_popularity(u, train_df, pop_rank, k=max(ks))
        for k in ks:
            metrics[k]["p"].append(precision_at_k(recs, truth, k))
            metrics[k]["r"].append(recall_at_k(recs, truth, k))
            metrics[k]["n"].append(ndcg_at_k(recs, truth, k))

    print("Popularity baseline (leave-last-out), users =", len(users))
    for k in ks:
        p = sum(metrics[k]["p"]) / len(metrics[k]["p"])
        r = sum(metrics[k]["r"]) / len(metrics[k]["r"])
        n = sum(metrics[k]["n"]) / len(metrics[k]["n"])
        print(f"K={k:>2}  Precision@K={p:.4f}  Recall@K={r:.4f}  NDCG@K={n:.4f}")

if __name__ == "__main__":
    main()
