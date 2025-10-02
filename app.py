import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import io, json

from model import SimpleFaceNet   # ✅ Import the same model used in training

app = FastAPI()

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFaceNet(num_classes=2)

try:
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)   # ✅ exact match now
    model.eval()
    print("✅ SimpleFaceNet loaded successfully!")
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
# Evaluation Endpoint
# -------------------------
@app.get("/evaluation")
def get_evaluation_results():
    try:
        with open("evaluation_results.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def root():
    return {"message": "Federated Learning API is running 🚀"}
