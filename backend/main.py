"""
FastAPI backend cho ứng dụng phân loại trạng thái trái cây.

Chạy:  uvicorn main:app --reload --port 8000
Cần đặt cạnh file này: fruit_model.h5 và class_names.json (do train_model.py sinh ra).
"""
import io
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fruit-api")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fruit_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(title="Fruit Ripeness Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Nạp mô hình một lần khi khởi động -----------------------
if not MODEL_PATH.exists():
    raise RuntimeError(f"Không tìm thấy mô hình: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if CLASS_NAMES_PATH.exists():
    CLASS_NAMES = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
else:
    # Thứ tự giống train_model.py
    CLASS_NAMES = ["fresh", "rotten"]
    logger.warning("Không có class_names.json, dùng thứ tự mặc định: %s", CLASS_NAMES)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Đọc ảnh, resize 224x224, chuẩn hóa giống lúc huấn luyện."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # kiểm tra file có phải ảnh thật
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="File tải lên không phải ảnh hợp lệ.")

    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)  # đưa về khoảng [-1, 1] như MobileNetV2
    return np.expand_dims(arr, axis=0)


def run_prediction(batch: np.ndarray):
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100


@app.get("/")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng '{file.content_type}' không được hỗ trợ. Hãy dùng JPG, PNG, WEBP hoặc BMP.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File rỗng.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 10 MB).")

    batch = preprocess_image(contents)

    try:
        label, confidence = await run_in_threadpool(run_prediction, batch)
    except Exception:
        logger.exception("Lỗi khi dự đoán")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi chạy mô hình.")

    return {"label": label, "confidence": round(confidence, 2)}
