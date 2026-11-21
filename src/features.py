"""
Trích xuất đặc trưng (feature engineering) từ ảnh để dùng với các thuật toán học máy cổ điển.

Mỗi ảnh được biến thành 1 vector số gồm N_FEATURES đặc trưng thủ công:
    1. Histogram màu HSV (Hue 16 bin, Saturation 8 bin, Value 8 bin)  -> 32 đặc trưng
       Trái cây tươi và hỏng khác nhau rõ về phân bố màu (hỏng thường nâu, tối, xỉn).
    2. Trung bình và độ lệch chuẩn của kênh R,G,B và H,S,V            -> 12 đặc trưng
    3. Tỉ lệ điểm ảnh tối và tỉ lệ điểm ảnh màu nâu                   ->  2 đặc trưng
    4. Texture: độ lớn gradient trung bình/độ lệch chuẩn theo x và y   ->  4 đặc trưng
       (vết thâm, đốm làm bề mặt kém đồng đều hơn).

Hue là góc tròn (đỏ nằm cả ở 0 lẫn 1), nên trục hue được dịch HUE_SHIFT để dải màu đỏ -> cam -> vàng -> xanh
nằm liền nhau; nhờ đó histogram, trung bình và độ lệch chuẩn của hue đúng với cả táo đỏ lẫn táo xanh.

Nếu USE_FOREGROUND_MASK = True, mọi đặc trưng chỉ tính trên phần QUẢ (không tính nền):
màu nền được ước lượng từ viền ảnh, điểm ảnh nào khác màu nền đủ nhiều thì là quả. Nhờ đó ảnh nền xanh,
nền xám... không còn làm lệch histogram so với ảnh train nền trắng. Không tách được nền thì dùng cả ảnh.

preprocess_image_bytes() được API dùng khi dự đoán và PHẢI cho ra đúng vector như lúc train.
"""
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.data_loader import CLASS_NAMES, IMG_SIZE, PROCESSED_DIR, ROOT_DIR, SPLIT_TAG, load_splits

H_BINS, S_BINS, V_BINS = 16, 8, 8

USE_FOREGROUND_MASK = True  # False = tính đặc trưng trên cả ảnh như bản cũ
FEATURE_VERSION = ("fg2" if USE_FOREGROUND_MASK else "full2")  # đổi cách tính -> cache đặc trưng tự tạo lại
HUE_SHIFT = 0.1    # dịch trục hue để màu đỏ (quanh 0 và 1) không bị cắt đôi

_BORDER = 3        # độ dày dải viền (pixel, trên ảnh 64x64) dùng để ước lượng màu nền
_BG_DIST = 0.18    # khoảng cách RGB (thang 0-1) tối thiểu so với màu nền để tính là quả
_BORDER_STD = 0.12 # viền lộn xộn hơn mức này (quả chạm mép ảnh) thì không tin màu nền

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


def foreground_mask(rgb):
    """
    rgb: mảng (H, W, 3) giá trị 0-1. Trả về mask bool (H, W) của phần quả,
    hoặc None nếu không tách được (viền không đồng nhất, hoặc mask quá nhỏ/quá lớn).
    """
    b = _BORDER
    border = np.concatenate(
        [rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    if np.linalg.norm(border - bg, axis=1).mean() > _BORDER_STD:
        return None
    mask = np.linalg.norm(rgb - bg, axis=2) > _BG_DIST
    frac = mask.mean()
    if frac < 0.05 or frac > 0.95:
        return None
    return mask


def extract_features(image: Image.Image) -> np.ndarray:
    """Ảnh PIL -> vector đặc trưng 1 chiều (N_FEATURES,)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    sel = foreground_mask(rgb) if USE_FOREGROUND_MASK else None
    if sel is None:
        sel = np.ones(rgb.shape[:2], dtype=bool)

    rgb_p, hsv_p = rgb[sel], hsv[sel]  # (n, 3): chỉ các điểm ảnh thuộc quả
    h_raw = hsv_p[:, 0]  # hue gốc (dùng cho brown_ratio)
    hsv_p = hsv_p.copy()
    hsv_p[:, 0] = (h_raw + HUE_SHIFT) % 1.0  # hue đã dịch: dùng cho histogram và thống kê
    h, s, v = hsv_p[:, 0], hsv_p[:, 1], hsv_p[:, 2]

    parts = []

    # 1. Histogram màu HSV (chuẩn hoá để tổng = 1, không phụ thuộc kích thước ảnh)
    for channel, bins in ((h, H_BINS), (s, S_BINS), (v, V_BINS)):
        hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
        parts.append(hist / hist.sum())

    # 2. Thống kê màu: mean và std của R,G,B rồi H,S,V
    parts.append(np.concatenate([rgb_p.mean(axis=0), hsv_p.mean(axis=0)]))
    parts.append(np.concatenate([rgb_p.std(axis=0), hsv_p.std(axis=0)]))

    # 3. Tỉ lệ vùng tối và vùng nâu (dấu hiệu của trái cây hỏng)
    dark_ratio = (v < 0.3).mean()
    brown_ratio = ((h_raw > 0.04) & (h_raw < 0.12) & (s > 0.3) & (v < 0.6)).mean()
    parts.append(np.array([dark_ratio, brown_ratio]))

    # 4. Texture: gradient theo hai trục trên ảnh xám (chỉ tính giữa các điểm ảnh cùng thuộc quả)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    ok_x = sel[:, 1:] & sel[:, :-1]
    ok_y = sel[1:, :] & sel[:-1, :]
    if ok_x.sum() >= 10 and ok_y.sum() >= 10:
        gx, gy = gx[ok_x], gy[ok_y]
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

    Trích xuất đặc trưng khá tốn thời gian nên kết quả được lưu ở
    data/processed/features_<tập>_<chế độ dữ liệu>_<cách tính đặc trưng>.npz và dùng lại ở các lần sau.
    Cache tự bị bỏ nếu danh sách ảnh của tập đó thay đổi.
    """
    train_df, val_df, test_df = load_splits()
    result = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        cache = os.path.join(PROCESSED_DIR, f"features_{name}_{SPLIT_TAG}_{FEATURE_VERSION}.npz")
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
