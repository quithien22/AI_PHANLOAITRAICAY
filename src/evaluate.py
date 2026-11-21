"""Đánh giá mô hình trên tập test và vẽ biểu đồ (lưu vào reports/)."""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from src.data_loader import CLASS_NAMES, REPORTS_DIR
from src.features import FEATURE_NAMES


def evaluate_model(model, X_test, y_test, out_dir=REPORTS_DIR, prefix=""):
    """
    Đánh giá trên tập test.
    Lưu {prefix}confusion_matrix.png và {prefix}classification_report.txt.
    Trả về dict {test_loss, test_acc}.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, y_prob, labels=list(range(len(CLASS_NAMES))))
    print(f"\nTest log-loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (tập test)")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}confusion_matrix.png"), dpi=150)
    plt.close(fig)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nBáo cáo phân loại:\n")
    print(report)
    with open(os.path.join(out_dir, f"{prefix}classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test log-loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}\n\n{report}")

    return {"test_loss": float(test_loss), "test_acc": float(test_acc)}


def plot_feature_importance(model, out_path=None, top=15):
    """
    Vẽ độ quan trọng của đặc trưng (chỉ với Random Forest / Decision Tree).
    Trả về đường dẫn ảnh, hoặc None nếu thuật toán không có feature_importances_.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        return None

    out_path = out_path or os.path.join(REPORTS_DIR, "feature_importance.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    order = clf.feature_importances_.argsort()[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * top + 1.5))
    ax.barh([FEATURE_NAMES[i] for i in order], clf.feature_importances_[order])
    ax.set_title(f"Top {top} đặc trưng quan trọng nhất")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
