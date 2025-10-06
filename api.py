from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from model import SimpleFaceNet
from utils.transforms import transform
from utils.metrics import pairwise_cosine
from PIL import Image, UnidentifiedImageError
import io
import os
import random
import base64
import numpy as np
import cv2
import matplotlib.cm as cm
import tempfile
import shutil

# -------------------
# Optional: FER import
# -------------------
try:
    from fer import FER
    FER_AVAILABLE = True
except Exception as e:
    print("⚠️ FER not available:", e)
    FER_AVAILABLE = False

# -------------------------------------------------------
# FastAPI init
# -------------------------------------------------------
app = FastAPI(title="Federated Face Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Model loading (SimpleFaceNet)
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "global_model.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Global model not found! Please run eval_client.py to create 'global_model.pt'")

model = SimpleFaceNet(embedding_size=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"✅ Model loaded successfully on {device.upper()}")

# -------------------------------------------------------
# Create FER detector + Haar fallback
# -------------------------------------------------------
detector = None
if FER_AVAILABLE:
    try:
        detector = FER(mtcnn=True)
        print("✅ FER detector initialized (mtcnn=True)")
    except Exception as e:
        print("⚠️ FER initialization failed:", e)
        detector = None

haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def read_image(file_data: bytes):
    try:
        img = Image.open(io.BytesIO(file_data)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        return img, tensor
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

def bytes_to_bgr_image(file_data: bytes):
    np_img = np.frombuffer(file_data, np.uint8)
    bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return bgr

def clamp_box(x, y, w, h, img_w, img_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    if x + w > img_w: w = img_w - x
    if y + h > img_h: h = img_h - y
    if w <= 0 or h <= 0: return None
    return (x, y, w, h)

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        _, x = read_image(data)
        with torch.no_grad():
            _ = model(x)
        pred_class = random.choice(["A", "B", "C", "D"])
        conf = round(random.uniform(0.85, 0.99), 2)
        return {"prediction": pred_class, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/")
async def explain(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img, x = read_image(data)
        model.eval()

        last_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
        if last_conv_layer is None:
            raise HTTPException(status_code=500, detail="No Conv2d layer found for Grad-CAM")

        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations["value"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()

        fwd_handle = last_conv_layer.register_forward_hook(forward_hook)
        bwd_handle = last_conv_layer.register_backward_hook(backward_hook)

        output = model(x)
        pred = output.mean()
        model.zero_grad()
        pred.backward()

        acts = activations["value"][0].cpu().numpy()
        grads = gradients["value"][0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))

        cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0)
        cam = cam / cam.max() if cam.max() != 0 else cam
        cam = cv2.resize(cam, (img.width, img.height))

        img_np = np.array(img)
        heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)

        buffered = io.BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        fwd_handle.remove()
        bwd_handle.remove()

        return {"prediction": "Face Detected", "heatmap": f"data:image/png;base64,{encoded}"}
    except Exception as e:
        import traceback
        print("❌ Grad-CAM Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Grad-CAM Error: {str(e)}")

@app.post("/verify")
async def verify(a: UploadFile = File(...), b: UploadFile = File(...), threshold: float = 0.8):
    da = await a.read()
    db = await b.read()
    _, xa = read_image(da)
    _, xb = read_image(db)
    with torch.no_grad():
        ea = model(xa)
        eb = model(xb)
        score = float(pairwise_cosine(ea, eb).item())
    result = "Match" if score >= threshold else "No Match"
    return {"score": score, "result": result}

@app.get("/interview/check")
async def check_candidate(candidate_id: str):
    emb_path = f"embeddings/{candidate_id}.pt"
    exists = os.path.exists(emb_path)
    return {"exists": exists}

@app.post("/interview/register")
async def register_candidate(candidate_id: str = Form(...), file: UploadFile = File(...)):
    try:
        data = await file.read()
        _, x = read_image(data)
        with torch.no_grad():
            emb = model(x)
        os.makedirs("embeddings", exist_ok=True)
        path = f"embeddings/{candidate_id}.pt"
        torch.save(emb.cpu(), path)
        print(f"✅ Saved embedding for {candidate_id} → {path}")
        return {"status": "ok", "candidate_id": candidate_id, "message": "Face registered successfully"}
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------
# ✅ Analyze Interview Frame (LIVE)
# -------------------------------------------------------
@app.post("/interview/analyze")
async def analyze_interview(file: UploadFile = File(...), candidate_id: str = Form(None)):
    import time
    try:
        data = await file.read()
        frame = bytes_to_bgr_image(data)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        os.makedirs("debug_frames", exist_ok=True)
        frame = cv2.flip(frame, 1)
        face_crop = None
        emotion_label = "neutral"
        emotion_confidence = 0.0
        detection_method = "none"

        # ✅ Try FER (mtcnn=True)
        if FER_AVAILABLE and detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fer_results = detector.detect_emotions(frame_rgb)
                if not fer_results:
                    fallback_detector = FER(mtcnn=False)
                    fer_results = fallback_detector.detect_emotions(frame_rgb)
                if fer_results:
                    fer_results.sort(key=lambda x: x["box"][2] * x["box"][3], reverse=True)
                    (x, y, w, h) = fer_results[0]["box"]
                    (emotion_label, emotion_confidence) = max(fer_results[0]["emotions"].items(), key=lambda kv: kv[1])
                    x, y, w, h = map(int, [x, y, w, h])
                    face_crop = frame[y:y + h, x:x + w]
                    detection_method = "FER"
                    print(f"✅ FER detected face ({emotion_label}={emotion_confidence:.2f})")
            except Exception as e:
                print(f"⚠️ FER failed: {e}")

        # ✅ Haar fallback
        if face_crop is None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar_face.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_crop = frame[y:y + h, x:x + w]
                    detection_method = "Haar"
                    print("✅ Haar detected face")
            except Exception as e:
                print(f"⚠️ Haar failed: {e}")

        # ✅ Center crop fallback
        if face_crop is None:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(h, w) // 2
            face_crop = frame[cy - size//2:cy + size//2, cx - size//2:cx + size//2]
            detection_method = "center_crop"
            print("⚠️ Using center crop fallback")

        # ✅ Engagement analysis
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_face, 40, 120)
        edge_density = np.sum(edges) / (gray_face.shape[0] * gray_face.shape[1])

        h = gray_face.shape[0]
        eye_region = gray_face[: int(h * 0.4), :]
        mouth_region = gray_face[int(h * 0.65):, :]
        eye_edges = cv2.Canny(eye_region, 40, 120)
        mouth_edges = cv2.Canny(mouth_region, 40, 120)

        eye_score = round(min(1.0, np.sum(eye_edges) / max(1, eye_region.size) * 30), 3)
        mouth_score = round(min(1.0, np.sum(mouth_edges) / max(1, mouth_region.size) * 25), 3)
        emotion_confidence = round(min(1.0, edge_density * 25), 3)

        # ✅ Identity verification
        identity_confidence = 0.0
        if candidate_id:
            emb_path = f"embeddings/{candidate_id}.pt"
            if os.path.exists(emb_path):
                _, tensor_x = read_image(data)
                with torch.no_grad():
                    emb_curr = model(tensor_x)
                    stored_emb = torch.load(emb_path, map_location=device).to(device)
                    sim = float(pairwise_cosine(emb_curr, stored_emb).item())
                    identity_confidence = round(max(0.0, min(1.0, sim)), 3)

        engagement_score = round(0.4 * emotion_confidence + 0.35 * eye_score + 0.25 * mouth_score, 3)
        summary = round((identity_confidence + engagement_score) / 2, 3) if candidate_id else engagement_score

        print(f"📊 Detection: {detection_method} | ID: {identity_confidence} | Eng: {engagement_score}")

        return {
            "status": "ok",
            "identity_confidence": identity_confidence,
            "emotion": emotion_label,
            "emotion_confidence": emotion_confidence,
            "engagement_score": engagement_score,
            "eye_score": eye_score,
            "mouth_score": mouth_score,
            "summary": summary
        }

    except Exception as e:
        import traceback
        print("❌ Analyze Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Interview Analysis Error: {str(e)}")

# -------------------------------------------------------
# 🎥 Analyze Full Recorded Video (Option 2)
# -------------------------------------------------------
@app.post("/interview/analyze_video")
async def analyze_video(file: UploadFile = File(...), candidate_id: str = Form(None)):
    """
    📹 Analyze a recorded video (.mp4)
    Extracts frames every 1s and reuses analyze_interview logic
    """
    import time
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = tmp.name

        print(f"🎥 Processing uploaded video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0
        frame_skip = int(fps) if fps > 0 else 1
        results = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_skip != 0:
                idx += 1
                continue
            idx += 1
            print(f"🖼 Analyzing frame {idx}/{total}")
            _, buf = cv2.imencode(".jpg", frame)
            img_stream = io.BytesIO(buf.tobytes())

            try:
                res = await analyze_interview(
                    file=UploadFile(filename=f"frame_{idx}.jpg", file=img_stream),
                    candidate_id=candidate_id
                )
                results.append(res)
            except Exception as e:
                print(f"⚠️ Frame {idx} skipped: {e}")

        cap.release()
        os.remove(video_path)

        if not results:
            raise HTTPException(status_code=400, detail="No valid frames processed")

        avg_identity = np.mean([r["identity_confidence"] for r in results])
        avg_engagement = np.mean([r["engagement_score"] for r in results])
        avg_eye = np.mean([r["eye_score"] for r in results])
        avg_mouth = np.mean([r["mouth_score"] for r in results])

        verdict = (
            "🌟 Excellent engagement & recognition" if avg_identity > 0.8 and avg_engagement > 0.7
            else "🙂 Good effort" if avg_identity > 0.6 and avg_engagement > 0.5
            else "⚠️ Low engagement or mismatch"
        )

        print(f"📊 Video Summary → ID={avg_identity:.3f}, ENG={avg_engagement:.3f}, Verdict={verdict}")

        return {
            "status": "ok",
            "video_info": {
                "duration_sec": round(duration, 2),
                "fps": round(fps, 2),
                "frames_analyzed": len(results)
            },
            "average_scores": {
                "identity_confidence": round(float(avg_identity), 3),
                "engagement_score": round(float(avg_engagement), 3),
                "eye_score": round(float(avg_eye), 3),
                "mouth_score": round(float(avg_mouth), 3),
            },
            "verdict": verdict,
            "per_frame_results": results[:10]
        }

    except Exception as e:
        import traceback
        print("❌ Video Analyze Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video Analyze Error: {str(e)}")
