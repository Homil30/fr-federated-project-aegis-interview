import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import io, json, base64
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import mediapipe as mp
import os

from model import SimpleFaceNet   # ✅ Import same model used in training

app = FastAPI()

# -------------------------
# ✅ Enable CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Explainability (Grad-CAM) + Predict + Evaluation
# -------------------------
# (your existing endpoints remain unchanged)
# ...

# ===========================================================
# INTERVIEW ANALYSIS FEATURE
# ===========================================================

mp_face = mp.solutions.face_mesh
_face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

REF_DIR = Path("interview_refs")
REF_DIR.mkdir(exist_ok=True)

def image_bytes_to_pil(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")

@torch.no_grad()
def get_embedding_from_pil(pil_img):
    model.eval()
    x = transform(pil_img).unsqueeze(0).to(device)
    if hasattr(model, "get_embedding"):
        return model.get_embedding(x).squeeze().cpu().numpy()
    with torch.no_grad():
        out = model(x).squeeze().cpu().numpy()
    return out

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def mediapipe_face_metrics(pil_img):
    img = np.array(pil_img)
    results = _face_mesh.process(img)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    def xy(i): return np.array([lm[i].x * w, lm[i].y * h])
    def dist(i, j): return np.linalg.norm(xy(i) - xy(j))

    left_eye = dist(159, 145)
    right_eye = dist(386, 374)
    mouth = dist(13, 14)
    face_width = dist(130, 359) + 1e-8

    eye_score = (left_eye + right_eye) / (2.0 * face_width)
    mouth_score = mouth / face_width
    return {"eye_score": float(eye_score), "mouth_score": float(mouth_score)}

@app.post("/interview/register")
async def interview_register(candidate_id: str = Form(...), file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        pil = image_bytes_to_pil(img_bytes)
        emb = get_embedding_from_pil(pil)
        np.save(REF_DIR / f"{candidate_id}.npy", emb)
        return {"status": "ok", "candidate_id": candidate_id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/interview/analyze")
async def interview_analyze(file: UploadFile = File(...), candidate_id: str = Form(None)):
    try:
        img_bytes = await file.read()
        pil = image_bytes_to_pil(img_bytes)

        identity_confidence = None
        if candidate_id:
            ref_path = REF_DIR / f"{candidate_id}.npy"
            if ref_path.exists():
                ref_emb = np.load(ref_path)
                emb = get_embedding_from_pil(pil)
                sim = cosine_similarity(emb, ref_emb)
                identity_confidence = float((sim + 1.0) / 2.0)

        metrics = mediapipe_face_metrics(pil)
        if metrics is None:
            return {"error": "no_face_detected"}

        eye = metrics["eye_score"]
        mouth = metrics["mouth_score"]
        engagement = max(0.0, min(1.0, 1.2 * eye - 0.5 * mouth + 0.2))

        return {
            "identity_confidence": identity_confidence,
            "engagement_score": round(float(engagement), 4),
            "eye_score": round(float(eye), 4),
            "mouth_score": round(float(mouth), 4)
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ NEW ROUTE — check if candidate is already registered
@app.get("/interview/check")
def check_candidate(candidate_id: str):
    """Check if candidate embedding file exists."""
    file_path = os.path.join("interview_refs", f"{candidate_id}.npy")
    exists = os.path.exists(file_path)
    return {"exists": exists}

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
