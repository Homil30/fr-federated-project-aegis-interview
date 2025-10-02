from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from model import SimpleFaceNet
from utils import transform, pairwise_cosine
from PIL import Image
import io
import os

app = FastAPI()
device = "cpu"
MODEL_PATH = "global_model.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Global model not found. Please run eval_client.py to create {MODEL_PATH}")

# ✅ load FaceNet backbone
model = SimpleFaceNet(embedding_size=128, pretrained=True, backbone="facenet")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x)
    return {"embedding": emb.squeeze(0).tolist()}

@app.post("/compare")
async def compare(a: UploadFile = File(...), b: UploadFile = File(...)):
    da = await a.read()
    db = await b.read()
    ia = Image.open(io.BytesIO(da)).convert("RGB")
    ib = Image.open(io.BytesIO(db)).convert("RGB")
    xa = transform(ia).unsqueeze(0)
    xb = transform(ib).unsqueeze(0)
    with torch.no_grad():
        ea = model(xa)
        eb = model(xb)
        score = float(pairwise_cosine(ea, eb).item())
    return {"score": score}
