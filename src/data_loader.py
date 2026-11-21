"""
Hằng số đường dẫn + đọc dữ liệu.

File này cố ý KHÔNG import thư viện nặng ở đầu file (pandas, sklearn nạp khi cần)
để API chỉ cần import hằng số mà không phải cài thêm thư viện train.

Quy ước dữ liệu (data/raw/train, data/raw/test):
    thư mục bắt đầu bằng "fresh"  -> nhãn "fresh"
    thư mục bắt đầu bằng "rotten" -> nhãn "rotten"

Dữ liệu gốc của Kaggle đã bị augment sẵn (rotated_by_30_..., translation_..., saltandpepper_...)
và các bản augment của cùng một ảnh nằm lẫn cả ở train lẫn test -> rò rỉ dữ liệu (data leakage).
Vì vậy mặc định (ORIGINALS_ONLY = True) chỉ giữ ẢNH GỐC, gộp train+test rồi chia lại theo ảnh gốc.
Các file trong data/raw KHÔNG bị xoá hay sửa.
"""
import os
import re

# ----------------------------- Đường dẫn -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
TRAIN_DIR = os.path.join(RAW_DIR, "train")
TEST_DIR = os.path.join(RAW_DIR, "test")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
EXPERIMENTS_CSV = os.path.join(REPORTS_DIR, "experiments.csv")
BEST_CONFIG_PATH = os.path.join(REPORTS_DIR, "best_experiment.json")

# ----------------------------- Hằng số dữ liệu -----------------------------
IMG_SIZE = (64, 64)  # ảnh được thu nhỏ về cỡ này trước khi trích xuất đặc trưng
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES = ["fresh", "rotten"]  # thứ tự này được lưu vào models/class_names.json
VAL_SIZE = 0.15   # tỉ lệ tập validation trên toàn bộ ảnh gốc
TEST_SIZE = 0.15  # tỉ lệ tập test trên toàn bộ ảnh gốc
SEED = 42

ORIGINALS_ONLY = True  # True: chỉ dùng ảnh gốc, bỏ các bản augment có sẵn trong data/raw
SPLIT_TAG = "original" if ORIGINALS_ONLY else "all"  # tên file chia dữ liệu khác nhau theo chế độ, không ghi đè nhau

# Tiền tố của ảnh augment trong dataset: rotated_by_15_, translation_, saltandpepper_, vertical_flip_...
_AUG_NAMED = re.compile(r"^(rotated_by_\d+|translation|saltandpepper|vertical_flip|horizontal_flip)_", re.IGNORECASE)
# Phòng khi có kiểu augment khác: bất kỳ tiền tố nào đứng trước "Screen Shot ..."
_AUG_GENERIC = re.compile(r"^[a-z0-9_]+_(?=screen shot)", re.IGNORECASE)


def is_original(fname):
    """True nếu tên file là ảnh gốc (không có tiền tố augment)."""
    return not (_AUG_NAMED.match(fname) or _AUG_GENERIC.match(fname))


def base_name(fname):
    """Tên ảnh gốc của một file (bỏ tiền tố augment). Mọi bản augment của 1 ảnh có chung base_name."""
    fname = _AUG_NAMED.sub("", fname, count=1)
    return _AUG_GENERIC.sub("", fname, count=1)


def get_label(folder_name):
    """Gộp nhãn theo tên thư mục. Trả về None nếu không nhận ra."""
    name = folder_name.lower()
    if name.startswith("fresh"):
        return "fresh"
    if name.startswith("rotten"):
        return "rotten"
    return None


def build_dataframe(root, originals_only=None):
    """Quét thư mục -> DataFrame (filename, label, source). Mặc định bỏ ảnh augment (xem ORIGINALS_ONLY)."""
    import pandas as pd

    if originals_only is None:
        originals_only = ORIGINALS_ONLY

    rows, skipped = [], 0
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        label = get_label(folder)
        if label is None:
            print(f"Bỏ qua thư mục không nhận ra nhãn: {folder}")
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(IMG_EXT):
                continue
            if originals_only and not is_original(fname):
                skipped += 1
                continue
            rows.append(
                {
                    "filename": os.path.join(folder_path, fname),
                    "label": label,
                    "source": folder,  # dùng để chia tầng theo từng loại trái cây
                }
            )
    if not rows:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong: {root}")
    if originals_only:
        print(f"{os.path.basename(root)}: giữ {len(rows)} ảnh gốc, bỏ qua {skipped} ảnh augment")
    return pd.DataFrame(rows)


RAW_INDEX_CSV = os.path.join(INTERIM_DIR, f"raw_index_{SPLIT_TAG}.csv")
SPLIT_CSVS = {name: os.path.join(PROCESSED_DIR, f"{name}_{SPLIT_TAG}.csv") for name in ("train", "val", "test")}


def _to_rel(df):
    out = df.copy()
    out["filename"] = [os.path.relpath(p, ROOT_DIR) for p in out["filename"]]
    return out


def _to_abs(df):
    out = df.copy()
    out["filename"] = [os.path.join(ROOT_DIR, p) for p in out["filename"]]
    return out


def load_splits(val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SEED, refresh=False):
    """
    Trả về (train_df, val_df, test_df).

    Gộp ảnh của data/raw/train và data/raw/test, rồi chia lại theo NHÓM (cùng loại trái + cùng ảnh gốc),
    chia tầng theo loại trái cây. Nhờ vậy mọi bản augment của một ảnh luôn nằm cùng một tập, không rò rỉ
    giữa train / val / test.

    - data/interim/raw_index_<tag>.csv : danh mục toàn bộ ảnh đã quét (trung gian)
    - data/processed/{train,val,test}_<tag>.csv : bản chia cố định, mọi thí nghiệm dùng chung
    Lần đầu sẽ tạo các file này; lần sau đọc lại. Đổi dữ liệu raw thì gọi load_splits(refresh=True).
    """
    import pandas as pd

    if not refresh and all(os.path.exists(p) for p in SPLIT_CSVS.values()):
        train_df, val_df, test_df = (_to_abs(pd.read_csv(SPLIT_CSVS[n])) for n in ("train", "val", "test"))
    else:
        from sklearn.model_selection import train_test_split

        full_df = pd.concat([build_dataframe(TRAIN_DIR), build_dataframe(TEST_DIR)], ignore_index=True)
        full_df["group"] = full_df["source"] + "/" + full_df["filename"].map(lambda p: base_name(os.path.basename(p)))

        groups = full_df.drop_duplicates("group")[["group", "source"]].reset_index(drop=True)
        g_trainval, g_test = train_test_split(
            groups, test_size=test_size, stratify=groups["source"], random_state=seed
        )
        g_train, g_val = train_test_split(
            g_trainval, test_size=val_size / (1.0 - test_size), stratify=g_trainval["source"], random_state=seed
        )

        def pick(g):
            return full_df[full_df["group"].isin(set(g["group"]))].drop(columns="group").reset_index(drop=True)

        train_df, val_df, test_df = pick(g_train), pick(g_val), pick(g_test)

        os.makedirs(INTERIM_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        index_df = pd.concat(
            [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")]
        )
        _to_rel(index_df).to_csv(RAW_INDEX_CSV, index=False, encoding="utf-8")
        for name, df in ("train", train_df), ("val", val_df), ("test", test_df):
            _to_rel(df).to_csv(SPLIT_CSVS[name], index=False, encoding="utf-8")

    print("Số ảnh theo nhãn:")
    print("  Train:", train_df["label"].value_counts().to_dict())
    print("  Val  :", val_df["label"].value_counts().to_dict())
    print("  Test :", test_df["label"].value_counts().to_dict())
    return train_df, val_df, test_df


if __name__ == "__main__":
    # python -m src.data_loader            -> tạo interim/ + processed/ (nếu chưa có)
    # python -m src.data_loader --refresh  -> tạo lại từ data/raw
    import sys

    load_splits(refresh="--refresh" in sys.argv)
    print("\nĐã tạo:")
    print(" ", RAW_INDEX_CSV)
    for _path in SPLIT_CSVS.values():
        print(" ", _path)
