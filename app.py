from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json

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
# Load Model
# -----------------------
MODEL_PATH = "global_model.pt"  # or global_model_round_3.pt

try:
    # Load ResNet18 backbone (because checkpoint has "features" & "classifier")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (Male/Female or Known/Unknown)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None

# -----------------------
# Image Preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet standard
])

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_tensor = transform(image).unsqueeze(0)

        if model is None:
            return {"error": "Model not loaded"}

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
