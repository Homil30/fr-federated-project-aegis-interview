from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json

# -----------------------
# Import your model class
# -----------------------
# ✅ Adjust import to match your project structure
from models.simple_facenet import SimpleFaceNet  

# -----------------------
# Setup FastAPI
# -----------------------
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Model (state_dict)
# -----------------------
MODEL_PATH = "global_model.pt"

try:
    # ✅ Rebuild model with correct architecture
    model = SimpleFaceNet(num_classes=2)  # ⚠️ change num_classes if needed
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image (⚠️ adjust normalization to match training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # adjust if trained differently
        ])
        img_tensor = transform(image).unsqueeze(0)

        if model is None:
            return {"error": "Model not loaded"}

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return {"prediction": int(predicted.item())}

    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Evaluation Endpoint
# -----------------------
@app.get("/evaluation")
def get_evaluation_results():
    try:
        with open("evaluation_results.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Root Endpoint
# -----------------------
@app.get("/")
def root():
    return {"message": "Aegis Federated Learning API is running 🚀"}
