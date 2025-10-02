import torch
from model import SimpleFaceNet   # ✅ Import training model

# Init model
model = SimpleFaceNet(num_classes=2)

# Load checkpoint (from federated training)
checkpoint_path = "global_model_round_3.pt"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Load into model
model.load_state_dict(state_dict, strict=True)

# Save cleaned weights for FastAPI
torch.save(model.state_dict(), "model.pth")

print("✅ Successfully converted checkpoint -> model.pth")
