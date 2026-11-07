"""
Huấn luyện mô hình phân loại trạng thái trái cây (Fresh / Rotten)
bằng Transfer Learning với MobileNetV2.

Dataset đọc trực tiếp từ cấu trúc hiện có, KHÔNG cần di chuyển file:
    dataset/
        train/  freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges
        test/   (cấu trúc tương tự)

Thư mục bắt đầu bằng "fresh" -> nhãn "fresh", bắt đầu bằng "rotten" -> nhãn "rotten".
(Khi có thêm ảnh trái cây xanh, bạn thêm thư mục "unripe..." và mở rộng hàm get_label.)

Chạy:  python train_model.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------- Cấu hình -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
VAL_SIZE = 0.2
SEED = 42
MODEL_PATH = os.path.join(BASE_DIR, "fruit_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CLASS_NAMES = ["fresh", "rotten"]  # thứ tự này cũng được lưu cho backend


def get_label(folder_name):
    """Gộp nhãn theo tên thư mục. Trả về None nếu không nhận ra."""
    name = folder_name.lower()
    if name.startswith("fresh"):
        return "fresh"
    if name.startswith("rotten"):
        return "rotten"
    return None


def build_dataframe(root):
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


# ------------------------- Đọc dữ liệu & chia train/val -------------------------
full_train_df = build_dataframe(TRAIN_DIR)
test_df = build_dataframe(TEST_DIR)

train_df, val_df = train_test_split(
    full_train_df,
    test_size=VAL_SIZE,
    stratify=full_train_df["source"],
    random_state=SEED,
)

print("\nSố ảnh theo nhãn:")
print("  Train:", train_df["label"].value_counts().to_dict())
print("  Val  :", val_df["label"].value_counts().to_dict())
print("  Test :", test_df["label"].value_counts().to_dict())

# ------------------- Data Augmentation + Data Generator -------------------
# preprocess_input của MobileNetV2 đưa pixel về khoảng [-1, 1].
# Backend PHẢI dùng đúng hàm này để tiền xử lý.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode="nearest",
)

# Val/Test chỉ tiền xử lý, KHÔNG augmentation
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

common = dict(
    x_col="filename",
    y_col="label",
    classes=CLASS_NAMES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
val_gen = eval_datagen.flow_from_dataframe(val_df, shuffle=False, **common)
test_gen = eval_datagen.flow_from_dataframe(test_df, shuffle=False, **common)  # False: để ma trận nhầm lẫn đúng thứ tự

num_classes = len(CLASS_NAMES)
print("Các nhãn:", train_gen.class_indices)

# Lưu thứ tự nhãn để backend dùng lại (tránh lệch nhãn)
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False)

# ------------------------------ Xây dựng mô hình ------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # đóng băng các lớp cơ sở

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),  # lớp phân loại cuối
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ------------------------------- Huấn luyện -------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# --------------------------- Biểu đồ Loss / Accuracy ---------------------------
def plot_history(hist, out_path=os.path.join(BASE_DIR, "training_history.png")):
    acc, val_acc = hist.history["accuracy"], hist.history["val_accuracy"]
    loss, val_loss = hist.history["loss"], hist.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs_range, acc, label="Train accuracy")
    axes[0].plot(epochs_range, val_acc, label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, loss, label="Train loss")
    axes[1].plot(epochs_range, val_loss, label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


plot_history(history)

# ----------------------- Đánh giá trên tập TEST + Ma trận nhầm lẫn -----------------------
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

test_gen.reset()
y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("Confusion Matrix (tập test)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

print("\nBáo cáo phân loại:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ------------------------------- Lưu mô hình -------------------------------
model.save(MODEL_PATH)
print(f"Đã lưu mô hình: {MODEL_PATH}")
print(f"Đã lưu danh sách nhãn: {CLASS_NAMES_PATH}")
