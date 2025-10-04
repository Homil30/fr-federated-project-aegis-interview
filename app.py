import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
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

@app.get("/")
def root():
    return {"message": "Federated Learning API is running 🚀"}
