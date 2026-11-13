"""
Hằng số đường dẫn + đọc dữ liệu.

File này cố ý KHÔNG import thư viện nặng ở đầu file (pandas, sklearn nạp khi cần)
để API chỉ cần import hằng số mà không phải cài thêm thư viện train.

Quy ước dữ liệu (data/raw/train, data/raw/test):
    thư mục bắt đầu bằng "fresh"  -> nhãn "fresh"
    thư mục bắt đầu bằng "rotten" -> nhãn "rotten"
"""
import os

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
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
EXPERIMENTS_CSV = os.path.join(REPORTS_DIR, "experiments.csv")
BEST_CONFIG_PATH = os.path.join(REPORTS_DIR, "best_experiment.json")

# ----------------------------- Hằng số dữ liệu -----------------------------
IMG_SIZE = (224, 224)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES = ["fresh", "rotten"]  # thứ tự này được lưu vào models/class_names.json
VAL_SIZE = 0.2
SEED = 42


def get_label(folder_name):
    """Gộp nhãn theo tên thư mục. Trả về None nếu không nhận ra."""
    name = folder_name.lower()
    if name.startswith("fresh"):
        return "fresh"
    if name.startswith("rotten"):
        return "rotten"
    return None


def build_dataframe(root):
    """Quét thư mục -> DataFrame (filename, label, source)."""
    import pandas as pd

    rows = []
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        label = get_label(folder)
        if label is None:
            print(f"Bỏ qua thư mục không nhận ra nhãn: {folder}")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(IMG_EXT):
                rows.append(
                    {
                        "filename": os.path.join(folder_path, fname),
                        "label": label,
                        "source": folder,  # dùng để chia tầng theo từng loại trái cây
                    }
                )
    if not rows:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong: {root}")
    return pd.DataFrame(rows)


RAW_INDEX_CSV = os.path.join(INTERIM_DIR, "raw_index.csv")
SPLIT_CSVS = {name: os.path.join(PROCESSED_DIR, f"{name}.csv") for name in ("train", "val", "test")}


def _to_rel(df):
    out = df.copy()
    out["filename"] = [os.path.relpath(p, ROOT_DIR) for p in out["filename"]]
    return out


def _to_abs(df):
    out = df.copy()
    out["filename"] = [os.path.join(ROOT_DIR, p) for p in out["filename"]]
    return out


def load_splits(val_size=VAL_SIZE, seed=SEED, refresh=False):
    """
    Trả về (train_df, val_df, test_df).

    - data/interim/raw_index.csv : danh mục toàn bộ ảnh quét từ data/raw (trung gian)
    - data/processed/{train,val,test}.csv : bản chia cố định, mọi thí nghiệm dùng chung
    Lần đầu sẽ tạo các file này; lần sau đọc lại. Đổi dữ liệu raw thì gọi load_splits(refresh=True).
    """
    import pandas as pd

    if not refresh and all(os.path.exists(p) for p in SPLIT_CSVS.values()):
        train_df, val_df, test_df = (_to_abs(pd.read_csv(SPLIT_CSVS[n])) for n in ("train", "val", "test"))
    else:
        from sklearn.model_selection import train_test_split

        full_train_df = build_dataframe(TRAIN_DIR)
        test_df = build_dataframe(TEST_DIR)
        train_df, val_df = train_test_split(
            full_train_df,
            test_size=val_size,
            stratify=full_train_df["source"],
            random_state=seed,
        )
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

        os.makedirs(INTERIM_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        index_df = pd.concat([full_train_df.assign(split="train"), test_df.assign(split="test")])
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
