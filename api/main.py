"""
REST API phân loại trạng thái trái cây (fresh / rotten).

Chỉ cung cấp API (không có giao diện web). Test bằng Postman, xem docs/.

Chạy từ thư mục gốc dự án:
    uvicorn api.main:app --reload --port 8000

Swagger tự động: http://localhost:8000/docs
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from src.data_loader import BEST_MODEL_PATH, CLASS_NAMES_PATH
from src.features import InvalidImageError, preprocess_image_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fruit-api")

MODEL_PATH = Path(BEST_MODEL_PATH)  # models/best_model.h5 (do src.experiments / src.train sinh ra)
CLASS_NAMES_PATH = Path(CLASS_NAMES_PATH)  # models/class_names.json
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# Danh sách origin được phép gọi API từ trình duyệt, phân tách bằng dấu phẩy. Mặc định: cho tất cả.
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(
    title="Fruit Ripeness Classifier API",
    version="1.1.0",
    description="Nhận ảnh trái cây, trả về nhãn fresh/rotten và độ tin cậy.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=ALLOW_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Nạp mô hình một lần khi khởi động -----------------------
if not MODEL_PATH.exists():
    raise RuntimeError(f"Không tìm thấy mô hình: {MODEL_PATH}. Hãy chạy notebook 03 hoặc `python -m src.experiments` trước.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if CLASS_NAMES_PATH.exists():
    CLASS_NAMES = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
else:
    CLASS_NAMES = ["fresh", "rotten"]  # cùng thứ tự với src/data_loader.py
    logger.warning("Không có class_names.json, dùng thứ tự mặc định: %s", CLASS_NAMES)


def run_prediction(batch: np.ndarray) -> dict:
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probs)},
    }


# --------------------------------- Endpoints ---------------------------------
@app.get("/health", tags=["system"])
def health():
    """Kiểm tra API còn sống và model đã được nạp."""
    return {"status": "ok", "model_loaded": True}


@app.get("/classes", tags=["system"])
def classes():
    """Danh sách nhãn mà model có thể trả về."""
    return {"classes": CLASS_NAMES}


@app.post("/predict", tags=["predict"])
async def predict(file: UploadFile = File(...)):
    """Upload 1 ảnh (form-data, key = `file`) -> trả về nhãn và độ tin cậy (%)."""
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

    try:
        batch = preprocess_image_bytes(contents)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await run_in_threadpool(run_prediction, batch)
    except Exception:
        logger.exception("Lỗi khi dự đoán")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi chạy mô hình.")

    return result
