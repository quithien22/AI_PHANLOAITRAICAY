"""
Chạy nhiều thuật toán học máy, so sánh và CHỐT thuật toán tốt nhất.

Quy tắc chọn: val_acc cao nhất; nếu bằng nhau thì val_loss (log-loss) thấp hơn.
Chỉ dùng tập VALIDATION để chọn. Tập test chỉ dùng 1 lần để báo cáo kết quả cuối của model thắng,
tránh chọn model "vừa khít" với tập test.

Đầu ra:
    reports/experiments.csv            bảng so sánh tất cả thí nghiệm
    reports/experiments_comparison.png biểu đồ so sánh
    reports/best_experiment.json       cấu hình + chỉ số của thí nghiệm thắng
    reports/confusion_matrix.png, classification_report.txt  (của model thắng)
    models/best_model.joblib, models/class_names.json         (API dùng)

Chạy:  python -m src.experiments
"""
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import BEST_CONFIG_PATH, EXPERIMENTS_CSV, REPORTS_DIR
from src.evaluate import evaluate_model, plot_feature_importance
from src.features import load_feature_splits
from src.train import fit_model, save_artifacts

EXPERIMENTS = [
    {"name": "E1_logistic_regression", "algo": "logistic_regression", "params": {"C": 1.0}},
    {"name": "E2_knn_k5", "algo": "knn", "params": {"n_neighbors": 5}},
    {"name": "E3_naive_bayes", "algo": "naive_bayes", "params": {}},
    {"name": "E4_decision_tree_depth10", "algo": "decision_tree", "params": {"max_depth": 10}},
    {"name": "E5_random_forest_200", "algo": "random_forest", "params": {"n_estimators": 200}},
    {"name": "E6_svm_rbf_C10", "algo": "svm", "params": {"kernel": "rbf", "C": 10.0, "gamma": "scale"}},
]


def plot_comparison(df, out_path):
    d = df.sort_values("val_acc")
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(d) + 1.5))
    bars = ax.barh(d["name"], d["val_acc"] * 100)
    for bar, (_, row) in zip(bars, d.iterrows()):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {row['val_acc'] * 100:.2f}% (loss {row['val_loss']:.3f})",
            va="center",
        )
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title("So sánh các thuật toán")
    lo = max(0, d["val_acc"].min() * 100 - 5)
    ax.set_xlim(lo, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiments(names=None, refresh_features=False):
    """
    Chạy các thí nghiệm trong EXPERIMENTS (hoặc chỉ những tên trong `names`).
    Trả về (DataFrame kết quả đã sắp xếp, dict thí nghiệm thắng).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = load_feature_splits(refresh=refresh_features)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    rows, best = [], None
    for cfg in EXPERIMENTS:
        if names and cfg["name"] not in names:
            continue
        print(f"\n{'=' * 70}\n>>> {cfg['name']}\n{'=' * 70}")

        t0 = time.time()
        model, val = fit_model(cfg, X_train, y_train, X_val, y_val)
        row = {
            "name": cfg["name"],
            "algo": cfg["algo"],
            "params": json.dumps(cfg["params"]),
            **val,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"val_acc={val['val_acc']:.4f}  val_loss={val['val_loss']:.4f}  ({row['seconds']}s)")
        rows.append(row)
        pd.DataFrame(rows).to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8")  # lưu dần, lỡ dừng giữa chừng

        if best is None or (val["val_acc"], -val["val_loss"]) > (best["row"]["val_acc"], -best["row"]["val_loss"]):
            best = {"model": model, "cfg": cfg, "row": row}

    if best is None:
        raise RuntimeError("Không có thí nghiệm nào được chạy (kiểm tra tham số `names`).")

    results = pd.DataFrame(rows).sort_values(["val_acc", "val_loss"], ascending=[False, True]).reset_index(drop=True)
    results.to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8")
    plot_comparison(results, os.path.join(REPORTS_DIR, "experiments_comparison.png"))

    # Chốt: đánh giá model thắng trên test (1 lần), lưu model + cấu hình
    print(f"\n>>> THẮNG: {best['cfg']['name']}")
    test = evaluate_model(best["model"], X_test, y_test)
    plot_feature_importance(best["model"])
    save_artifacts(best["model"])

    info = {
        "selection_rule": "val_acc cao nhất, hòa thì val_loss thấp hơn (chỉ dùng tập validation)",
        "config": best["cfg"],
        "val": {k: best["row"][k] for k in ("val_acc", "val_loss", "val_f1")},
        "test": test,
    }
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {BEST_CONFIG_PATH}")

    return results, info


if __name__ == "__main__":
    run_experiments()
