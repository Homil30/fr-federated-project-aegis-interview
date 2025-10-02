import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import io, json

app = FastAPI()

# -------------------------
# Correct GlobalModel (matches checkpoint: features.0,3,6,9 and classifier.1,4)
# -------------------------
class GlobalModel(nn.Module):
    def __init__(self, num_classes=2):
        super(GlobalModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # features.0
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # features.3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # features.6
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# features.9
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(12544, 512),  # classifier.1
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),  # classifier.4
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)   # Flatten for linear layer
        x = self.classifier(x)
        return x


# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlobalModel(num_classes=2)

try:
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ GlobalModel loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None


# -------------------------
# Image Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        if model is None:
            return {"error": "Model not loaded"}

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        return {
            "prediction": int(pred),
            "confidence": round(confidence, 4),
            "probabilities": {
                "class_0": round(probs[0].item(), 4),
                "class_1": round(probs[1].item(), 4)
            }
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def root():
    return {"message": "Federated Learning API is running 🚀"}
