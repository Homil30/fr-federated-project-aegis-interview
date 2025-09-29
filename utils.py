# utils.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# -------------------------------
# Image Preprocessing Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Load single image and preprocess
# -------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dimension

# -------------------------------
# Cosine similarity between embeddings
# -------------------------------
def pairwise_cosine(emb1, emb2):
    """
    emb1: (N, d) tensor
    emb2: (M, d) tensor
    returns: cosine similarity matrix (N x M)
    """
    emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2 = emb2 / emb2.norm(dim=1, keepdim=True)
    return torch.mm(emb1, emb2.t())
