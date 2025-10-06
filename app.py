# ===========================================================
# app.py — FINAL VERSION (with Improved Grad-CAM Heatmap)
# ===========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import io, base64, cv2, numpy as np
from pathlib import Path
import mediapipe as mp
from io import BytesIO
import os

from model import SimpleFaceNet

# ===========================================================
# FastAPI init + CORS
# ===========================================================
app = FastAPI(title="Aegis Federated Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# Load model
# ===========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = SimpleFaceNet(embedding_size=128).to(device)
    model.load_state_dict(torch.load("global_model.pt", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# ===========================================================
# Preprocessing
# ===========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===========================================================
# /predict/ endpoint
# ===========================================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = torch.argmax(torch.abs(outputs), dim=1).item()
            confidence = torch.norm(outputs, dim=1).item() / 128.0

        return {"prediction": int(pred_class), "confidence": round(float(confidence), 4)}
    except Exception as e:
        return {"error": str(e)}


# ===========================================================
# /explain/ endpoint (Improved Grad-CAM with Face-Focused Enhancement)
# ===========================================================
@app.post("/explain/")
async def explain(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # 1️⃣ Load & preprocess image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # 2️⃣ Find last Conv layer
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise HTTPException(status_code=500, detail="No Conv2D layer found in model.")

        fmap, grads = {}, {}

        def forward_hook(module, inp, out):
            fmap["value"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            grads["value"] = grad_out[0].detach()

        fh = last_conv.register_forward_hook(forward_hook)
        bh = last_conv.register_full_backward_hook(backward_hook)

        # 3️⃣ Forward + Backward pass
        outputs = model(input_tensor)
        pred_class = int(torch.argmax(torch.abs(outputs), dim=1).item())
        target_score = outputs.norm(p=2)
        model.zero_grad()
        target_score.backward()

        fh.remove()
        bh.remove()

        # 4️⃣ Generate Grad-CAM heatmap
        gradients = grads["value"].cpu().numpy()[0]
        activations = fmap["value"].cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        # Resize to match original image
        cam_resized = cv2.resize(cam, (pil_image.width, pil_image.height))

        # 5️⃣ Enhance focus on face region (suppress background noise)
        cam_resized = cv2.GaussianBlur(cam_resized, (9, 9), 2)
        cam_power = np.power(cam_resized, 1.4)  # amplify mid activations
        cam_power = np.clip(cam_power / cam_power.max(), 0, 1)

        # Suppress background by adaptive thresholding
        thresh = np.percentile(cam_power, 35)
        cam_refined = np.where(cam_power > thresh, cam_power, 0)
        cam_refined = cv2.GaussianBlur(cam_refined, (9, 9), 1)
        cam_refined = cam_refined / (cam_refined.max() + 1e-8)

        # 6️⃣ Convert to vivid heatmap (TURBO colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_refined), cv2.COLORMAP_TURBO)

        # Color tuning: more red/yellow, less blue
        hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 1.3     # saturation boost
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.15 + 10, 0, 255)
        heatmap = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 7️⃣ Overlay heatmap on original image (balanced blending)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.7, 0)

        # 8️⃣ Local contrast & clarity enhancement
        overlay = cv2.convertScaleAbs(overlay, alpha=1.18, beta=8)
        sharpen = cv2.GaussianBlur(overlay, (0, 0), 1.2)
        overlay = cv2.addWeighted(overlay, 1.25, sharpen, -0.25, 0)

        # 9️⃣ Encode to base64
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 96])
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {
            "prediction": pred_class,
            "heatmap": f"data:image/jpeg;base64,{img_str}"
        }

    except Exception as e:
        import traceback
        print("Error in /explain/:")
        print(traceback.format_exc())
        return {"error": str(e)}


# ===========================================================
# Interview Features (unchanged)
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
    return model(x).squeeze().cpu().numpy()

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

@app.post("/interview/register")
async def interview_register(candidate_id: str = Form(...), file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        pil = image_bytes_to_pil(img_bytes)
        emb = get_embedding_from_pil(pil)
        np.save(REF_DIR / f"{candidate_id}.npy", emb)
        return {"status": "ok", "candidate_id": candidate_id, "message": "Face registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/interview/verify")
async def verify_identity(file: UploadFile = File(...), candidate_id: str = Form(...)):
    try:
        img_bytes = await file.read()
        pil = image_bytes_to_pil(img_bytes)
        emb = get_embedding_from_pil(pil)

        ref_path = REF_DIR / f"{candidate_id}.npy"
        if not ref_path.exists():
            return {"verified": False, "reason": "Reference not found"}

        ref_emb = np.load(ref_path)
        sim = cosine_similarity(emb, ref_emb)
        return {"verified": sim > 0.7, "similarity": float(sim)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Aegis Federated Face Recognition API is running"}
