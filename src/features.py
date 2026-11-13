"""
Tiền xử lý & tăng cường dữ liệu ảnh.

preprocess_image_bytes() được API dùng khi dự đoán và PHẢI khớp với pipeline lúc train:
resize 224x224 + preprocess_input của MobileNetV2 (pixel về khoảng [-1, 1]).
"""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.data_loader import IMG_SIZE

# Các mức tăng cường dữ liệu dùng trong thí nghiệm (chỉ áp dụng cho tập train)
AUGMENTATIONS = {
    "normal": dict(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    ),
    "strong": dict(
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.35,
        brightness_range=(0.6, 1.4),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    ),
}


class InvalidImageError(ValueError):
    """File tải lên không phải ảnh hợp lệ."""


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Bytes ảnh -> mảng (1, 224, 224, 3) sẵn sàng đưa vào model."""
    try:
        Image.open(io.BytesIO(image_bytes)).verify()  # kiểm tra file có phải ảnh thật
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("File tải lên không phải ảnh hợp lệ.") from exc

    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)
