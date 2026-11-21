"""
Huấn luyện mô hình học máy phân loại fresh/rotten trên vector đặc trưng ảnh (xem src/features.py).

- Các hàm build_model / fit_model được notebooks và experiments.py dùng lại.
- Chạy trực tiếp sẽ huấn luyện lại bằng cấu hình tốt nhất trong reports/best_experiment.json
  (nếu chưa chạy thí nghiệm thì dùng DEFAULT_CONFIG), rồi lưu models/best_model.joblib:

    python -m src.train

Một cấu hình (cfg) gồm: {"name": ..., "algo": <khoá trong ALGORITHMS>, "params": {siêu tham số}}.
"""
import json
import os

import joblib
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import BEST_CONFIG_PATH, BEST_MODEL_PATH, CLASS_NAMES, CLASS_NAMES_PATH, MODELS_DIR, SEED
from src.evaluate import evaluate_model
from src.features import load_feature_splits

# Các thuật toán học máy được dùng (mỗi mục nhận dict siêu tham số và trả về bộ phân loại)
ALGORITHMS = {
    "logistic_regression": lambda p: LogisticRegression(max_iter=1000, random_state=SEED, **p),
    "knn": lambda p: KNeighborsClassifier(**p),
    "naive_bayes": lambda p: GaussianNB(**p),
    "decision_tree": lambda p: DecisionTreeClassifier(random_state=SEED, **p),
    "random_forest": lambda p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
    "svm": lambda p: SVC(probability=True, random_state=SEED, **p),
}

# Cấu hình baseline: Logistic Regression (mô hình đơn giản nhất)
DEFAULT_CONFIG = {
    "name": "E1_logistic_regression",
    "algo": "logistic_regression",
    "params": {"C": 1.0},
}


def build_model(cfg):
    """
    Pipeline gồm 2 bước: chuẩn hoá (StandardScaler) rồi bộ phân loại.
    Chuẩn hoá giúp KNN, SVM, Logistic Regression không bị lệch bởi đặc trưng có thang đo lớn.
    """
    algo = cfg["algo"]
    if algo not in ALGORITHMS:
        raise ValueError(f"algo không hợp lệ: {algo}. Chọn một trong {list(ALGORITHMS)}")
    classifier = ALGORITHMS[algo](cfg.get("params", {}))
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def fit_model(cfg, X_train, y_train, X_val, y_val):
    """Huấn luyện trên tập train, đo trên tập val. Trả về (model, {"val_acc", "val_loss", "val_f1"})."""
    model = build_model(cfg)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    metrics = {
        "val_acc": float(accuracy_score(y_val, y_pred)),
        "val_loss": float(log_loss(y_val, y_prob, labels=list(range(len(CLASS_NAMES))))),
        "val_f1": float(f1_score(y_val, y_pred, average="macro")),
    }
    return model, metrics


def save_artifacts(model):
    """Lưu models/best_model.joblib và models/class_names.json (API đọc 2 file này)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False)
    print(f"Đã lưu mô hình: {BEST_MODEL_PATH}")
    print(f"Đã lưu danh sách nhãn: {CLASS_NAMES_PATH}")


def load_best_config():
    """Cấu hình thắng cuộc từ reports/best_experiment.json; không có (hoặc là file cũ của bản deep learning) thì dùng DEFAULT_CONFIG."""
    if os.path.exists(BEST_CONFIG_PATH):
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f).get("config", {})
        if cfg.get("algo") in ALGORITHMS:
            return cfg
    return DEFAULT_CONFIG


def train_final(cfg=None):
    """Huấn luyện bằng cấu hình đã chọn, đánh giá trên val và test, lưu model."""
    cfg = cfg or load_best_config()
    print("Cấu hình huấn luyện:", cfg)

    data = load_feature_splits()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    model, val = fit_model(cfg, X_train, y_train, X_val, y_val)
    print("Validation:", val)

    test = evaluate_model(model, X_test, y_test)
    save_artifacts(model)
    return {"config": cfg, "val": val, "test": test}


if __name__ == "__main__":
    train_final()
