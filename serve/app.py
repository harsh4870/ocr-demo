import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
REVIEW_BELOW = float(os.environ.get("REVIEW_BELOW", "0.9"))

STATES = set("AP AR AS BR CG CH DL GA GJ HP HR JH JK KA KL LA MH ML MN MP MZ NL OD PB PY "
             "RJ SK TN TR TS UK UP WB".split())
STANDARD = re.compile(r"([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})")
BHARAT = re.compile(r"(\d{2})(BH)(\d{4})([A-Z]{1,2})")

meta = json.loads((MODEL_DIR / "model.json").read_text())
CHARSET, HEIGHT, MEAN, STD = meta["charset"], meta["height"], meta["mean"], meta["std"]
opts = ort.SessionOptions()
opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
SAMPLES = sorted(Path(os.environ.get("SAMPLES_DIR", MODEL_DIR / "samples")).glob("*.jpg"))

app = FastAPI(title="OCR API", version="1.0")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    w = max(16, round(img.width * HEIGHT / img.height))
    x = np.asarray(img.resize((w, HEIGHT), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((x - MEAN) / STD)[None, None]


def decode(logits: np.ndarray) -> tuple[str, float]:
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    best, conf = probs.argmax(-1), probs.max(-1)
    text, confs, prev = [], [], 0
    for i, c in zip(best, conf):
        if i != prev and i != 0:
            text.append(CHARSET[i])
            confs.append(c)
        prev = i
    return "".join(text), float(np.mean(confs)) if confs else 0.0


def check_plate(text: str) -> tuple[str, bool]:
    if m := BHARAT.fullmatch(text):
        return " ".join(m.groups()), True
    if m := STANDARD.fullmatch(text):
        return " ".join(m.groups()), m[1] in STATES
    return text, False


def recognize(img: Image.Image) -> dict:
    t0 = time.perf_counter()
    logits = session.run(None, {"image": preprocess(img)})[0][0]
    text, confidence = decode(logits)
    plate, valid = check_plate(text)
    return {"text": text, "plate": plate, "valid_format": valid, "confidence": round(confidence, 4),
            "needs_review": not valid or confidence < REVIEW_BELOW,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": os.environ.get("MODEL_VERSION", "dev"),
            "charset_size": len(CHARSET) - 1}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 2 MB")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "image dimensions too large")
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a valid image")
    return recognize(img)


@app.get("/sample")
def sample():
    if not SAMPLES:
        raise HTTPException(404, "no sample images bundled (see SAMPLES_DIR)")
    f = random.choice(SAMPLES)
    data = f.read_bytes()
    return {"label": f.name.split("_")[0], "image": base64.b64encode(data).decode(),
            **recognize(Image.open(io.BytesIO(data)))}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
