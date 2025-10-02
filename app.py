import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import io, json, base64
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleFaceNet   # ✅ Import same model used in training

app = FastAPI()

# -------------------------
# ✅ Enable CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend (port 5500) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFaceNet(num_classes=2).to(device)

try:
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
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
# Explainability (Grad-CAM) Endpoint
# -------------------------
@app.post("/explain/")
async def explain(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        if model is None:
            return {"error": "Model not loaded"}

        # ✅ Hook the last convolutional layer dynamically
        gradients = []
        activations = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Find last Conv2d
        target_layer = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                target_layer = layer

        if target_layer is None:
            return {"error": "No Conv2d layer found in model for Grad-CAM"}

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        # Forward + Backward
        outputs = model(input_tensor)
        pred_class = outputs.argmax(dim=1).item()
        score = outputs[0, pred_class]
        model.zero_grad()
        score.backward()

        # Compute Grad-CAM
        grads = gradients[0].detach().cpu().numpy()[0]     # [C,H,W]
        acts = activations[0].detach().cpu().numpy()[0]    # [C,H,W]
        weights = grads.mean(axis=(1, 2))                  # [C]

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # Resize CAM to image size
        cam_img = np.uint8(255 * cam)
        cam_img = np.stack([cam_img]*3, axis=-1)  # convert to RGB-like

        # Overlay heatmap
        plt.imshow(image)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        return {"prediction": pred_class, "heatmap": img_b64}

    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def root():
    return {"message": "Federated Learning API is running 🚀"}
