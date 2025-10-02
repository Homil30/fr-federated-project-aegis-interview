# inspect_checkpoint.py
import torch

checkpoint_path = "global_model_round_3.pt"
state_dict = torch.load(checkpoint_path, map_location="cpu")

print("\n📦 FEATURES")
for key, val in state_dict.items():
    if key.startswith("features"):
        print(f"{key:30s} {tuple(val.shape)}")

print("\n🧠 CLASSIFIER")
for key, val in state_dict.items():
    if key.startswith("classifier"):
        print(f"{key:30s} {tuple(val.shape)}")
