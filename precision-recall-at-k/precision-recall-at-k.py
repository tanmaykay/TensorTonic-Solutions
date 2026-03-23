def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k=recommended[:k]
    relevant_set=set(relevant)
    hits=0
    for i in top_k:
        if i in relevant_set:
            hits+=1
    precision=hits/k
    recall=hits/len(relevant)
    return [precision,recall]