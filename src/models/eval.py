import math

def precision_at_k(recs, truth, k: int):
    recs = recs[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for r in recs if r in truth)
    return hits / k

def recall_at_k(recs, truth, k: int):
    recs = recs[:k]
    if len(truth) == 0:
        return 0.0
    hits = sum(1 for r in recs if r in truth)
    return hits / len(truth)

def ndcg_at_k(recs, truth, k: int):
    recs = recs[:k]
    dcg = 0.0
    for idx, r in enumerate(recs, start=1):
        if r in truth:
            dcg += 1.0 / math.log2(idx + 1)
    # ideal DCG with |truth| hits at the top
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
