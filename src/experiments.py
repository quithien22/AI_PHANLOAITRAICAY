"""
Chạy nhiều thí nghiệm, so sánh và CHỐT thí nghiệm tốt nhất.

Quy tắc chọn: val_acc cao nhất; nếu bằng nhau thì val_loss thấp hơn.
Chỉ dùng tập VALIDATION để chọn. Tập test chỉ dùng 1 lần để báo cáo kết quả cuối của model thắng,
tránh chọn model "vừa khít" với tập test.

Đầu ra:
    reports/experiments.csv            bảng so sánh tất cả thí nghiệm
    reports/experiments_comparison.png biểu đồ so sánh
    reports/best_experiment.json       cấu hình + chỉ số của thí nghiệm thắng
    reports/training_history.png, confusion_matrix.png, classification_report.txt  (của model thắng)
    models/best_model.h5, models/class_names.json                                  (API dùng)

Chạy:  python -m src.experiments
"""
import gc
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data_loader import BEST_CONFIG_PATH, EXPERIMENTS_CSV, REPORTS_DIR, load_splits
from src.evaluate import evaluate_model, plot_history
from src.train import fit_model, make_generators, save_artifacts

_COMMON = {"dropout": 0.3, "dense_units": 128, "aug": "normal", "finetune_layers": 0, "finetune_lr": 1e-5}

EXPERIMENTS = [
    {**_COMMON, "name": "E1_cnn_scratch", "arch": "cnn", "lr": 1e-3},
    {**_COMMON, "name": "E2_mnv2_frozen_lr1e-3", "arch": "mobilenetv2", "lr": 1e-3},
    {**_COMMON, "name": "E3_mnv2_frozen_lr1e-4", "arch": "mobilenetv2", "lr": 1e-4},
    {**_COMMON, "name": "E4_mnv2_finetune_last30", "arch": "mobilenetv2", "lr": 1e-3, "finetune_layers": 30},
    {**_COMMON, "name": "E5_mnv2_strong_aug_dropout0.5", "arch": "mobilenetv2", "lr": 1e-3, "dropout": 0.5, "aug": "strong"},
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
    ax.set_title("So sánh các thí nghiệm")
    lo = max(0, d["val_acc"].min() * 100 - 5)
    ax.set_xlim(lo, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiments(epochs=15, names=None):
    """
    Chạy các thí nghiệm trong EXPERIMENTS (hoặc chỉ những tên trong `names`).
    Trả về (DataFrame kết quả đã sắp xếp, dict thí nghiệm thắng).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    train_df, val_df, test_df = load_splits()
    gen_cache = {}  # (train_gen, val_gen, test_gen) theo từng mức augmentation

    rows, best = [], None
    for cfg in EXPERIMENTS:
        if names and cfg["name"] not in names:
            continue
        print(f"\n{'=' * 70}\n>>> {cfg['name']}\n{'=' * 70}")

        if cfg["aug"] not in gen_cache:
            gen_cache[cfg["aug"]] = make_generators(train_df, val_df, test_df, cfg["aug"])
        train_gen, val_gen, _ = gen_cache[cfg["aug"]]

        t0 = time.time()
        model, history, val = fit_model(cfg, train_gen, val_gen, epochs)
        row = {
            **cfg,
            **val,
            "train_acc_last": history["accuracy"][-1],
            "epochs_run": len(history["loss"]),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8")  # lưu dần, lỡ dừng giữa chừng

        if best is None or (val["val_acc"], -val["val_loss"]) > (best["row"]["val_acc"], -best["row"]["val_loss"]):
            best = {"model": model, "history": history, "cfg": cfg, "row": row}
        else:
            del model
        gc.collect()

    if best is None:
        raise RuntimeError("Không có thí nghiệm nào được chạy (kiểm tra tham số `names`).")

    results = pd.DataFrame(rows).sort_values(["val_acc", "val_loss"], ascending=[False, True]).reset_index(drop=True)
    results.to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8")
    plot_comparison(results, os.path.join(REPORTS_DIR, "experiments_comparison.png"))

    # Chốt: đánh giá model thắng trên test (1 lần), lưu model + cấu hình
    print(f"\n>>> THẮNG: {best['cfg']['name']}")
    _, _, test_gen = gen_cache[best["cfg"]["aug"]]
    plot_history(best["history"])
    test = evaluate_model(best["model"], test_gen)
    save_artifacts(best["model"])

    info = {
        "selection_rule": "val_acc cao nhất, hòa thì val_loss thấp hơn (chỉ dùng tập validation)",
        "epochs_per_experiment": epochs,
        "config": {**best["cfg"], "epochs": 30},
        "val": {"val_acc": best["row"]["val_acc"], "val_loss": best["row"]["val_loss"]},
        "test": test,
    }
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {BEST_CONFIG_PATH}")

    tf.keras.backend.clear_session()
    return results, info


if __name__ == "__main__":
    run_experiments()
