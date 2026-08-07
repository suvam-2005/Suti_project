"""
backend/main.py — FastAPI inference server for the Disaster Detection classifier.

Loads the exported TorchScript model (model_scripted.pt) so it needs no
reference to train.py's model class — fully portable, drop it anywhere.
Works with however many classes were present at training time (reads
labels.json rather than hardcoding class names).

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health              -> liveness check + class list
    POST /predict  (multipart file="image")
         -> {"label": "wildfire", "confidence": 0.97, "probs": {...}}
"""

import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_scripted.pt"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
IMG_SIZE = 224

app = FastAPI(title="Disaster Detection API", version="1.0.0")

# Allow the upload UI (any origin) to call this API. Tighten allow_origins
# to your actual frontend domain(s) before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
labels = {}

preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.on_event("startup")
def load_model():
    global model, labels
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run train.py first to produce artifacts/model_scripted.pt"
        )
    model = torch.jit.load(str(MODEL_PATH), map_location=device)
    model.eval()
    labels = json.loads(LABELS_PATH.read_text())
    print(f"Loaded Disaster Detection model on {device}. Classes: {list(labels.values())}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": labels}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    result = {
        "label": labels[str(pred_idx)],
        "confidence": round(float(probs[pred_idx]), 4),
        "probs": {labels[str(i)]: round(float(p), 4) for i, p in enumerate(probs)},
    }
    return result
