# create_model.py
import torch
from model import SimpleFaceNet

# Create an untrained model that matches your architecture
model = SimpleFaceNet(embedding_size=128)

# Save only its weights as a state_dict
torch.save(model.state_dict(), "global_model.pt")

print("✅ Created new compatible global_model.pt successfully!")
    