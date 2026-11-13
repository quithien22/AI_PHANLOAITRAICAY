"""
Huấn luyện mô hình phân loại fresh/rotten.

- Các hàm build_model / fit_model / make_generators được notebooks và experiments.py dùng lại.
- Chạy trực tiếp sẽ huấn luyện lại bằng cấu hình tốt nhất trong reports/best_experiment.json
  (nếu chưa chạy thí nghiệm thì dùng DEFAULT_CONFIG), rồi lưu models/best_model.h5:

    python -m src.train
"""
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data_loader import (
    BEST_CONFIG_PATH,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    MODELS_DIR,
    SEED,
    load_splits,
)
from src.evaluate import evaluate_model, plot_history
from src.features import AUGMENTATIONS

BATCH_SIZE = 32

# Cấu hình baseline: MobileNetV2 đóng băng, lr 1e-3 (chính là mô hình trước đây của dự án)
DEFAULT_CONFIG = {
    "name": "E2_mnv2_frozen_lr1e-3",
    "arch": "mobilenetv2",
    "lr": 1e-3,
    "dropout": 0.3,
    "dense_units": 128,
    "aug": "normal",
    "finetune_layers": 0,
    "finetune_lr": 1e-5,
    "epochs": 30,
}


def make_generators(train_df, val_df, test_df, aug="normal"):
    """Tạo generator. Chỉ train có augmentation; val/test shuffle=False."""
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **AUGMENTATIONS[aug])
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
    test_gen = eval_datagen.flow_from_dataframe(test_df, shuffle=False, **common)
    return train_gen, val_gen, test_gen


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_model(cfg):
    """Trả về (model, base_model). base_model là None với kiến trúc CNN tự xây."""
    n = len(CLASS_NAMES)
    head = [
        layers.Dropout(cfg["dropout"]),
        layers.Dense(cfg["dense_units"], activation="relu"),
        layers.Dropout(cfg["dropout"]),
        layers.Dense(n, activation="softmax"),
    ]

    if cfg["arch"] == "cnn":
        base = None
        model = models.Sequential(
            [
                layers.Input(shape=IMG_SIZE + (3,)),
                layers.Conv2D(32, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(128, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.GlobalAveragePooling2D(),
                *head,
            ]
        )
    elif cfg["arch"] == "mobilenetv2":
        base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
        base.trainable = False
        model = models.Sequential([base, layers.GlobalAveragePooling2D(), *head])
    else:
        raise ValueError(f"arch không hợp lệ: {cfg['arch']}")

    _compile(model, cfg["lr"])
    return model, base


def _callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def fit_model(cfg, train_gen, val_gen, epochs=None):
    """
    Huấn luyện theo cấu hình. Nếu cfg["finetune_layers"] > 0: sau giai đoạn đầu (base đóng băng)
    mở khoá N lớp cuối của MobileNetV2 và train tiếp với lr nhỏ.

    Trả về (model, history_dict, {"val_loss", "val_acc"}).
    """
    epochs = epochs or cfg.get("epochs", 30)
    tf.keras.utils.set_random_seed(SEED)
    model, base = build_model(cfg)

    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=_callbacks(), verbose=1)
    history = {k: list(v) for k, v in hist.history.items()}

    n_unfreeze = cfg.get("finetune_layers", 0)
    if base is not None and n_unfreeze > 0:
        base.trainable = True
        for layer in base.layers[:-n_unfreeze]:
            layer.trainable = False
        _compile(model, cfg["finetune_lr"])  # bắt buộc compile lại sau khi đổi trainable
        hist2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=max(epochs // 2, 5),
            callbacks=_callbacks(),
            verbose=1,
        )
        for k, v in hist2.history.items():
            history.setdefault(k, []).extend(v)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)  # trọng số tốt nhất đã được khôi phục
    return model, history, {"val_loss": float(val_loss), "val_acc": float(val_acc)}


def save_artifacts(model):
    """Lưu models/best_model.h5 và models/class_names.json (API đọc 2 file này)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(BEST_MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False)
    print(f"Đã lưu mô hình: {BEST_MODEL_PATH}")
    print(f"Đã lưu danh sách nhãn: {CLASS_NAMES_PATH}")


def load_best_config():
    """Cấu hình thắng cuộc từ reports/best_experiment.json, nếu chưa có thì dùng DEFAULT_CONFIG."""
    if os.path.exists(BEST_CONFIG_PATH):
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)["config"]
    return DEFAULT_CONFIG


def train_final(cfg=None, epochs=None):
    """Huấn luyện đầy đủ bằng cấu hình đã chọn, đánh giá trên test, lưu model."""
    cfg = cfg or load_best_config()
    print("Cấu hình huấn luyện:", cfg)

    train_df, val_df, test_df = load_splits()
    train_gen, val_gen, test_gen = make_generators(train_df, val_df, test_df, cfg["aug"])

    model, history, val = fit_model(cfg, train_gen, val_gen, epochs)
    print("Validation:", val)

    plot_history(history)
    test = evaluate_model(model, test_gen)
    save_artifacts(model)
    return {"config": cfg, "val": val, "test": test}


if __name__ == "__main__":
    train_final()
