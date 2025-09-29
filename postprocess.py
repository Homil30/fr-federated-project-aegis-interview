# postprocess.py
import numpy as np

def fair_score_normalization(scores, groups):
    """
    Apply Fair Score Normalization (FSN).
    - Normalize similarity scores per group to have zero mean and unit variance.
    - Then rescale all groups back to a common distribution.
    Input:
        scores: numpy array of similarity scores
        groups: numpy array of demographic labels (same length as scores)
    Returns:
        normalized_scores: numpy array of adjusted scores
    """
    scores = np.array(scores)
    groups = np.array(groups)
    normalized_scores = np.zeros_like(scores, dtype=float)

    unique_groups = np.unique(groups)
    global_mean = scores.mean()
    global_std = scores.std() + 1e-8  # avoid divide by zero

    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        group_scores = scores[mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std() + 1e-8

        # Standardize within group
        standardized = (group_scores - group_mean) / group_std

        # Rescale to global distribution
        normalized = standardized * global_std + global_mean
        normalized_scores[mask] = normalized

    return normalized_scores
