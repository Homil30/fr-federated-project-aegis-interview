# eval_client.py
import argparse
import torch

from model import SimpleFaceNet as MyModel  # ✅ fixed import
from utils.save_load import load_global_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="saved_models/global_model_round1.pt")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

def main():
    model = MyModel()
    model = load_global_model(model, args.model_path, device=args.device)
    # Simple sanity check
    s = sum(p.sum().item() for p in model.parameters())
    print("[eval_client] Model param sum (sanity):", s)

if __name__ == "__main__":
    main()
