"""Đánh giá mô hình trên tập test và vẽ biểu đồ (lưu vào reports/)."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import CLASS_NAMES, REPORTS_DIR


def plot_history(history, out_path=None):
    """Vẽ accuracy/loss theo epoch. `history` là dict (history.history)."""
    out_path = out_path or os.path.join(REPORTS_DIR, "training_history.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    acc, val_acc = history["accuracy"], history["val_accuracy"]
    loss, val_loss = history["loss"], history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (train_v, val_v, title) in zip(
        axes, [(acc, val_acc, "Accuracy"), (loss, val_loss, "Loss")]
    ):
        ax.plot(epochs_range, train_v, label=f"Train {title.lower()}")
        ax.plot(epochs_range, val_v, label=f"Validation {title.lower()}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_model(model, test_gen, out_dir=REPORTS_DIR, prefix=""):
    """
    Đánh giá trên tập test. test_gen phải có shuffle=False.
    Lưu {prefix}confusion_matrix.png và {prefix}classification_report.txt.
    Trả về dict {test_loss, test_acc}.
    """
    os.makedirs(out_dir, exist_ok=True)

    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (tập test)")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}confusion_matrix.png"), dpi=150)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nBáo cáo phân loại:\n")
    print(report)
    with open(os.path.join(out_dir, f"{prefix}classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}\n\n{report}")

    return {"test_loss": float(test_loss), "test_acc": float(test_acc)}
