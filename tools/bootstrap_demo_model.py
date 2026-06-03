from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight offline demo model using the bundled synthetic dataset."
    )
    parser.add_argument("--train-dir", default="dataset/train", help="Training dataset root")
    parser.add_argument("--val-dir", default="dataset/val", help="Validation dataset root")
    parser.add_argument("--output-model", default="models/deepfake_model.h5", help="Output model path")
    parser.add_argument("--metrics-out", default="metrics/latest_metrics.json", help="Metrics JSON output path")
    parser.add_argument("--img-size", type=int, default=224, help="Input image side length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_datasets(
    train_dir: Path,
    val_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Expected train and val directories with class folders: real/ and fake/."
        )

    class_names = ["real", "fake"]

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)


def build_model(image_size: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(16, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="demo_deepfake_cnn",
    )


def collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[int] = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy().reshape(-1).astype("int32").tolist())
    return np.array(labels, dtype="int32")


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict[str, float | dict[str, int]]:
    y_pred = (y_scores >= threshold).astype("int32")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    accuracy = float((tp + tn) / total)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = float((2.0 * precision * recall) / (precision + recall))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, dict[str, float | dict[str, int]], list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_scores, best_threshold)

    for threshold in np.linspace(0.10, 0.90, 81):
        current_threshold = float(round(float(threshold), 2))
        metrics = compute_metrics(y_true, y_scores, current_threshold)
        results.append(
            {
                "threshold": current_threshold,
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
            }
        )

        current_key = (
            float(metrics["accuracy"]),
            float(metrics["f1_score"]),
            -abs(current_threshold - 0.5),
        )
        best_key = (
            float(best_metrics["accuracy"]),
            float(best_metrics["f1_score"]),
            -abs(best_threshold - 0.5),
        )
        if current_key > best_key:
            best_threshold = current_threshold
            best_metrics = metrics

    return best_threshold, best_metrics, results


def write_metrics(
    output_path: Path,
    dataset_name: str,
    base_threshold: float,
    best_threshold: float,
    base_metrics: dict[str, float | dict[str, int]],
    best_metrics: dict[str, float | dict[str, int]],
    threshold_scan: list[dict[str, float]],
    sample_count: int,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "sample_count": sample_count,
        "class_mapping": {"real": 0, "fake": 1},
        "threshold": base_threshold,
        "best_threshold": best_threshold,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "auc": None,
        "confusion_matrix": best_metrics["confusion_matrix"],
        "base_threshold_metrics": base_metrics,
        "threshold_scan": threshold_scan,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "target_80_achieved": float(best_metrics["accuracy"]) >= 0.80,
        "notes": "Offline demo model trained from the bundled synthetic sample dataset. Replace with a real dataset for project use.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_model = Path(args.output_model)
    metrics_out = Path(args.metrics_out)
    image_size = (args.img_size, args.img_size)

    train_ds, val_ds = load_datasets(train_dir, val_dir, image_size, args.batch_size, args.seed)

    model = build_model(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print("Training offline demo model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    y_true = collect_labels(val_ds)
    raw_predictions = model.predict(val_ds, verbose=0).astype("float32").reshape(-1)

    base_threshold = 0.5
    base_metrics = compute_metrics(y_true, raw_predictions, base_threshold)
    best_threshold, best_metrics, threshold_scan = find_best_threshold(y_true, raw_predictions)
    write_metrics(
        output_path=metrics_out,
        dataset_name=val_dir.name,
        base_threshold=base_threshold,
        best_threshold=best_threshold,
        base_metrics=base_metrics,
        best_metrics=best_metrics,
        threshold_scan=threshold_scan,
        sample_count=int(len(y_true)),
    )

    print("\nBootstrap complete")
    print(f"Model saved to: {output_model.resolve()}")
    print(f"Metrics saved to: {metrics_out.resolve()}")
    print(f"Best validation accuracy: {float(best_metrics['accuracy']) * 100:.2f}%")
    print(f"Best threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
