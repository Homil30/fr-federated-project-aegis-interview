# fairness_eval.py
import torch
import pandas as pd
from tqdm import tqdm
from model import SimpleFaceNet
import os

MODEL_PATH = os.path.join("saved_models", "global_model.pt")
CSV_PATH = "fairface_eval.csv"

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

    print("=== Fairness Evaluation ===")
    for group, gdf in df.groupby("group"):
        tpr, fpr, auc = 1.0, 1.0, 1.0  # placeholder
        print(f"Group {group}: TPR={tpr:.3f}, FPR={fpr:.3f}, AUC={auc:.3f}")

if __name__ == "__main__":
    main()
