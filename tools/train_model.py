from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BACKBONES = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetb2": keras.applications.EfficientNetB2,
    "efficientnetb3": keras.applications.EfficientNetB3,
    "efficientnetb4": keras.applications.EfficientNetB4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deepfake classifier (image-based).")
    parser.add_argument("--train-dir", default="dataset/train", help="Training dataset root")
    parser.add_argument("--val-dir", default="dataset/val", help="Validation dataset root")
    parser.add_argument("--output-model", default="models/deepfake_model.h5", help="Output model path")
    parser.add_argument("--history-out", default="metrics/train_history.json", help="Training history output path")
    parser.add_argument("--img-size", type=int, default=300, help="Image side size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Initial frozen-base epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=8, help="Additional fine-tuning epochs")
    parser.add_argument("--backbone", choices=sorted(BACKBONES.keys()), default="efficientnetb3", help="Feature extractor")
    parser.add_argument("--weights", default="imagenet", help="Backbone weights to load")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout before classification head")
    parser.add_argument("--dense-units", type=int, default=256, help="Hidden units in the classification head")
    parser.add_argument("--fine-tune-layers", type=int, default=48, help="How many backbone layers to unfreeze")
    parser.add_argument("--label-smoothing", type=float, default=0.02, help="Label smoothing for binary loss")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision on supported hardware")
    parser.add_argument("--crop-to-aspect-ratio", action="store_true", help="Crop instead of stretching source images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_model(
    image_size: int,
    backbone_name: str,
    weights: str,
    dropout: float,
    dense_units: int,
) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.12),
            layers.RandomBrightness(0.08),
            layers.GaussianNoise(0.01),
        ],
        name="augment",
    )

    backbone_builder = BACKBONES[backbone_name]
    base_model = backbone_builder(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if dense_units > 0:
        x = layers.Dense(dense_units, activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    else:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    model.base_model = base_model  # type: ignore[attr-defined]
    return model


def compile_model(model: keras.Model, learning_rate: float, label_smoothing: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def compute_class_weights(train_ds: tf.data.Dataset) -> dict[int, float]:
    real_count = 0
    fake_count = 0

    for _, labels in train_ds:
        flat = tf.reshape(labels, [-1]).numpy()
        real_count += int((flat == 0).sum())
        fake_count += int((flat == 1).sum())

    total = real_count + fake_count
    if real_count == 0 or fake_count == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: total / (2.0 * real_count),
        1: total / (2.0 * fake_count),
    }


def to_builtin(obj):
    if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
        return obj
    return float(obj)


def main() -> None:
    args = parse_args()
    resolved_weights = None if str(args.weights).lower() == "none" else args.weights

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    tf.keras.utils.set_random_seed(args.seed)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_model = Path(args.output_model)
    history_out = Path(args.history_out)

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Expected train and val directories with class folders: real/ and fake/."
        )

    image_shape = (args.img_size, args.img_size)

    # Force label mapping: real=0, fake=1
    class_names = ["real", "fake"]

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=image_shape,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        crop_to_aspect_ratio=args.crop_to_aspect_ratio,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=image_shape,
        batch_size=args.batch_size,
        shuffle=False,
        crop_to_aspect_ratio=args.crop_to_aspect_ratio,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    class_weights = compute_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    model = build_model(
        image_size=args.img_size,
        backbone_name=args.backbone,
        weights=resolved_weights,
        dropout=args.dropout,
        dense_units=args.dense_units,
    )
    compile_model(model, learning_rate=8e-4, label_smoothing=args.label_smoothing)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    history_out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(output_model, monitor="val_auc", mode="max", save_best_only=True),
    ]

    print("Training stage 1 (frozen backbone)...")
    history_stage_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    base_model = model.base_model  # type: ignore[attr-defined]
    base_model.trainable = True
    trainable_layers = max(1, min(len(base_model.layers), args.fine_tune_layers))
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-5, label_smoothing=args.label_smoothing)

    print("Training stage 2 (fine-tuning)...")
    history_stage_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs + args.fine_tune_epochs,
        initial_epoch=len(history_stage_1.history["loss"]),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.save(output_model)

    combined_history = {}
    for key, values in history_stage_1.history.items():
        combined_history[key] = list(values)
    for key, values in history_stage_2.history.items():
        combined_history.setdefault(key, []).extend(values)

    best_val_accuracy = max(combined_history.get("val_accuracy", [0.0]))
    best_val_auc = max(combined_history.get("val_auc", [0.0]))

    payload = {
        "backbone": args.backbone,
        "weights": resolved_weights,
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "crop_to_aspect_ratio": args.crop_to_aspect_ratio,
        "class_mapping": {"real": 0, "fake": 1},
        "class_weights": class_weights,
        "best_val_accuracy": best_val_accuracy,
        "best_val_auc": best_val_auc,
        "history": {k: to_builtin(v) for k, v in combined_history.items()},
    }
    history_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nTraining complete")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Best validation AUC     : {best_val_auc * 100:.2f}%")
    print(f"Model saved to: {output_model}")
    print(f"History saved to: {history_out}")


if __name__ == "__main__":
    main()
