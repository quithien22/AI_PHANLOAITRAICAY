"""
Trích xuất đặc trưng (feature engineering) từ ảnh để dùng với các thuật toán học máy cổ điển.

Mỗi ảnh được biến thành 1 vector số gồm N_FEATURES đặc trưng thủ công:
    1. Histogram màu HSV (Hue 16 bin, Saturation 8 bin, Value 8 bin)  -> 32 đặc trưng
       Trái cây tươi và hỏng khác nhau rõ về phân bố màu (hỏng thường nâu, tối, xỉn).
    2. Trung bình và độ lệch chuẩn của kênh R,G,B và H,S,V            -> 12 đặc trưng
    3. Tỉ lệ điểm ảnh tối và tỉ lệ điểm ảnh màu nâu                   ->  2 đặc trưng
    4. Texture: độ lớn gradient trung bình/độ lệch chuẩn theo x và y   ->  4 đặc trưng
       (vết thâm, đốm làm bề mặt kém đồng đều hơn).

preprocess_image_bytes() được API dùng khi dự đoán và PHẢI cho ra đúng vector như lúc train.
"""
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.data_loader import CLASS_NAMES, IMG_SIZE, PROCESSED_DIR, ROOT_DIR, load_splits

H_BINS, S_BINS, V_BINS = 16, 8, 8

FEATURE_NAMES = (
    [f"hist_H_{i}" for i in range(H_BINS)]
    + [f"hist_S_{i}" for i in range(S_BINS)]
    + [f"hist_V_{i}" for i in range(V_BINS)]
    + [f"{stat}_{ch}" for stat in ("mean", "std") for ch in ("R", "G", "B", "H", "S", "V")]
    + ["dark_ratio", "brown_ratio"]
    + ["grad_x_mean", "grad_y_mean", "grad_x_std", "grad_y_std"]
)
N_FEATURES = len(FEATURE_NAMES)  # 50


class InvalidImageError(ValueError):
    """File tải lên không phải ảnh hợp lệ."""


def extract_features(image: Image.Image) -> np.ndarray:
    """Ảnh PIL -> vector đặc trưng 1 chiều (N_FEATURES,)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    parts = []

    # 1. Histogram màu HSV (chuẩn hoá để tổng = 1, không phụ thuộc kích thước ảnh)
    for channel, bins in ((h, H_BINS), (s, S_BINS), (v, V_BINS)):
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
        parts.append(hist / hist.sum())

    # 2. Thống kê màu: mean và std của R,G,B rồi H,S,V
    parts.append(np.concatenate([rgb.mean(axis=(0, 1)), hsv.mean(axis=(0, 1))]))
    parts.append(np.concatenate([rgb.std(axis=(0, 1)), hsv.std(axis=(0, 1))]))

    # 3. Tỉ lệ vùng tối và vùng nâu (dấu hiệu của trái cây hỏng)
    dark_ratio = (v < 0.3).mean()
    brown_ratio = ((h > 0.04) & (h < 0.12) & (s > 0.3) & (v < 0.6)).mean()
    parts.append(np.array([dark_ratio, brown_ratio]))

    # 4. Texture: gradient theo hai trục trên ảnh xám
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    parts.append(np.array([gx.mean(), gy.mean(), gx.std(), gy.std()]))

    return np.concatenate(parts).astype(np.float32)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Bytes ảnh -> mảng (1, N_FEATURES) sẵn sàng đưa vào model."""
    try:
        Image.open(io.BytesIO(image_bytes)).verify()  # kiểm tra file có phải ảnh thật
        image = Image.open(io.BytesIO(image_bytes))
        features = extract_features(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("File tải lên không phải ảnh hợp lệ.") from exc
    return features.reshape(1, -1)


def _features_from_path(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return extract_features(img)


def build_feature_matrix(df, n_jobs=-1):
    """DataFrame (filename, label) -> (X, y). y là chỉ số lớp theo CLASS_NAMES (0 = fresh, 1 = rotten)."""
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(delayed(_features_from_path)(p) for p in df["filename"])
    X = np.vstack(rows)
    y = np.array([CLASS_NAMES.index(label) for label in df["label"]], dtype=np.int64)
    return X, y


def load_feature_splits(refresh=False, n_jobs=-1):
    """
    Trả về dict {"train": (X, y), "val": (X, y), "test": (X, y)}.

    Trích xuất đặc trưng khá tốn thời gian nên kết quả được lưu ở data/processed/features_<tập>.npz
    và dùng lại ở các lần sau. Cache tự bị bỏ nếu danh sách ảnh của tập đó thay đổi.
    """
    train_df, val_df, test_df = load_splits()
    result = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        cache = os.path.join(PROCESSED_DIR, f"features_{name}.npz")
        rel_names = np.array([os.path.relpath(p, ROOT_DIR) for p in df["filename"]])

        if not refresh and os.path.exists(cache):
            data = np.load(cache, allow_pickle=False)
            if data["X"].shape[1] == N_FEATURES and np.array_equal(data["files"], rel_names):
                result[name] = (data["X"], data["y"])
                continue

        print(f"Đang trích xuất đặc trưng tập {name} ({len(df)} ảnh)...")
        X, y = build_feature_matrix(df, n_jobs=n_jobs)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, files=rel_names)
        result[name] = (X, y)
    return result
