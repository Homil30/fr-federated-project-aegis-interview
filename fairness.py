# fairness.py
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_group_metrics(scores, labels, groups):
    """
    Compute per-group TPR, FPR, AUC.
    scores: numpy array of similarity scores
    labels: numpy array of 0/1 (same identity or not)
    groups: numpy array of demographic labels (strings)
    Returns: dict of {group: {TPR, FPR, AUC}}
    """
    results = {}
    unique_groups = np.unique(groups)

    # Pick a threshold (e.g., 0.5 for cosine similarity)
    threshold = 0.5

    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue

        g_scores = scores[mask]
        g_labels = labels[mask]

        # Predictions using threshold
        preds = (g_scores >= threshold).astype(int)

        TP = ((preds == 1) & (g_labels == 1)).sum()
        FP = ((preds == 1) & (g_labels == 0)).sum()
        TN = ((preds == 0) & (g_labels == 0)).sum()
        FN = ((preds == 0) & (g_labels == 1)).sum()

        TPR = TP / (TP + FN + 1e-8)  # avoid div by zero
        FPR = FP / (FP + TN + 1e-8)

        try:
            AUC = roc_auc_score(g_labels, g_scores)
        except:
            AUC = float("nan")

        results[g] = {"TPR": TPR, "FPR": FPR, "AUC": AUC}

    return results
