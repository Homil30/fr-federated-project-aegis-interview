import torch
from app import GlobalModel

# Init model with correct architecture
model = GlobalModel(num_classes=2)

# Load federated checkpoint
checkpoint_path = "global_model_round_3.pt"
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Load weights (strict=True because now it matches exactly)
model.load_state_dict(state_dict, strict=True)

# Save cleaned weights for FastAPI
torch.save(model.state_dict(), "model.pth")

print("✅ Successfully converted checkpoint -> model.pth")
