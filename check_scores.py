# check_scores.py
import torch
import pandas as pd
from tqdm import tqdm
from model import SimpleFaceNet
import os

MODEL_PATH = os.path.join("saved_models", "global_model.pt")
CSV_PATH = "fairface_eval.csv"

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # Load model
    model = SimpleFaceNet(embedding_size=128, pretrained=True, backbone="facenet")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load pairs
    df = pd.read_csv(CSV_PATH)

    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img1, img2, label = row["img1"], row["img2"], row["label"]
        # TODO: load and preprocess image properly
        # For now, assume pre-extracted embeddings or use dummy
        sims.append((label, 1.0))  # placeholder similarity

    df_sim = pd.DataFrame(sims, columns=["label", "similarity"])
    stats = df_sim.groupby("label")["similarity"].describe()
    print(stats)

if __name__ == "__main__":
    main()
