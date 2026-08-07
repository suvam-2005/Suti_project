# Disaster Detection

Image classifier that detects **flood, wildfire, cyclone, or earthquake** from a photo. Transfer-learning CNN → portable model export → FastAPI backend → upload UI.

```
disaster-detection/
├── README.md
├── train.py                  # training script (run this first)
├── data/                      # put your dataset here (see below)
│   ├── train/
│   │   ├── flood/
│   │   ├── wildfire/
│   │   ├── cyclone/
│   │   └── earthquake/
│   └── val/
│       ├── flood/
│       ├── wildfire/
│       ├── cyclone/
│       └── earthquake/
├── artifacts/                 # populated by train.py — model files land here
├── backend/
│   ├── main.py                # FastAPI inference server
│   └── requirements.txt
└── frontend/
    └── index.html             # upload UI, no build step needed
```

## 1. Dataset

**Kaggle: Cyclone, Wildfire, Flood, Earthquake Database**
https://www.kaggle.com/datasets/rupakroy/cyclone-wildfire-flood-earthquake-database

Has all four classes already separated into folders — this maps directly onto the `data/train/<class>` structure above, no relabeling needed (just lowercase the folder names to match: `Wildfire` → `wildfire`, etc.)

Backup / supplementary sources if you want more volume per class:
- Disasters Dataset (fire, flood, earthquake, neutral) — https://www.kaggle.com/datasets/georgemystriotis/disasters-dataset
- Disaster Images Dataset, 4,500 labeled images — search "Disaster Images Dataset CNN" on Kaggle

### How many images per class

| Split | Per class | Notes |
|---|---|---|
| Train | 600–1500 | 800–1000/class is a solid sweet spot for this problem |
| Val | 150–300 | ~20% of your train count |
| Test (optional) | 100–150 | Held out entirely from training, for a final sanity check |

4-class problems need somewhat more data per class than a 2-class one to get the same accuracy — below ~400/class expect the model to overfit faster, so lean toward the upper end of the range if you have it.

Auto-split a flat folder of images 80/20 into train/val:
```python
import os, shutil, random
src, cls = "raw_wildfire_images", "wildfire"
random.seed(42)
files = os.listdir(src)
random.shuffle(files)
split = int(len(files) * 0.8)
for f in files[:split]:
    shutil.copy(f"{src}/{f}", f"data/train/{cls}/{f}")
for f in files[split:]:
    shutil.copy(f"{src}/{f}", f"data/val/{cls}/{f}")
```

## 2. Train

```bash
pip install torch torchvision pillow
python train.py --data_dir data --epochs 15 --batch_size 32 --out_dir artifacts
```

- **ResNet18** pretrained on ImageNet, backbone frozen except the last block — fast to train, strong accuracy from day one since it already knows general visual features.
- Adapts automatically to however many class folders you have under `data/train/` — you're not locked to exactly 4 classes if you want to add "none/no-disaster" later.
- Early stopping (patience=5) + `ReduceLROnPlateau` scheduling built in.
- Expect **85–92% validation accuracy** with a clean, reasonably-sized dataset — 4-class disaster detection has more visual overlap than a 2-class problem (e.g. flood debris vs. earthquake rubble can look similar), so accuracy will typically land a bit lower than a pure flood-vs-fire model.

## 3. What gets produced (in `artifacts/`)

| File | Purpose |
|---|---|
| `best_model.pth` | PyTorch `state_dict` — for further training or raw weights |
| `model_scripted.pt` | **TorchScript** — self-contained, `torch.jit.load()`, this is what the backend uses |
| `model.onnx` | **ONNX** — runs anywhere: Python, C++, Java, mobile, or in-browser via `onnxruntime-web`. Hand this to anyone outside a Python/PyTorch stack. |
| `labels.json` | Index → class name mapping, e.g. `{"0":"cyclone","1":"earthquake","2":"flood","3":"wildfire"}` |
| `training_log.json` | Per-epoch loss/accuracy |

Zip the `artifacts/` folder and it's a complete, shippable model — no training code required to use it.

## 4. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/predict -F "image=@/path/to/test.jpg"
# -> {"label":"wildfire","confidence":0.94,"probs":{"cyclone":0.01,"earthquake":0.02,"flood":0.03,"wildfire":0.94}}
```

For production behind your existing infra: containerize this service, reverse-proxy `/api/detect` → `predict` through Nginx, and lock `allow_origins` down from `"*"` in `main.py` to your real frontend domain.

## 5. Run the UI

Open `frontend/index.html` directly in a browser — no build step, no dependencies. It posts to `http://localhost:8000/predict`; update the `API_URL` constant near the top of the `<script>` block once you deploy the backend for real.

The UI shows the top prediction plus a full confidence breakdown across all four classes, color-coded per disaster type (matches the legend at the top of the page).

## 6. Shipping the model elsewhere (outside your backend)

Hand someone `model.onnx` + `labels.json`. Minimal consumer with no PyTorch dependency:

```python
import onnxruntime as ort, numpy as np, json
from PIL import Image

labels = json.load(open("labels.json"))
sess = ort.InferenceSession("model.onnx")
img = Image.open("test.jpg").convert("RGB").resize((224, 224))
x = (np.asarray(img).astype("float32") / 255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
x = x.transpose(2,0,1)[None].astype("float32")
logits = sess.run(None, {"input": x})[0]
pred = labels[str(logits.argmax())]
print(pred)
```
