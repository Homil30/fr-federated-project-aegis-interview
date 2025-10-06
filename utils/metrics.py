import torch.nn.functional as F

def pairwise_cosine(a, b):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(a, b)
