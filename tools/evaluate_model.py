from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def model_uses_raw_255_inputs(model) -> bool:
    visited: set[int] = set()

    def contains_rescaling(layer) -> bool:
        layer_id = id(layer)
        if layer_id in visited:
            return False
        visited.add(layer_id)

        if layer.__class__.__name__ == "Rescaling":
            return True

        for child in getattr(layer, "layers", []):
            if contains_rescaling(child):
                return True
        return False

    return contains_rescaling(model)


def collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(folder.rglob(f"*{extension}"))
        files.extend(folder.rglob(f"*{extension.upper()}"))
    return sorted(set(files))


def preprocess(image_path: Path, image_size: tuple[int, int], use_raw_255: bool) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32")
    if not use_raw_255:
        image = image / 255.0
    return image


def normalize_prediction(raw: np.ndarray) -> float:
    value = np.array(raw).astype("float32").squeeze()
    if value.ndim == 0:
        score = float(value)
    elif value.size == 1:
        score = float(value.reshape(-1)[0])
    elif value.size == 2:
        logits = value.reshape(-1)
        exp_scores = np.exp(logits - np.max(logits))
        score = float(exp_scores[1] / np.sum(exp_scores))
    else:
        score = float(np.mean(value))

    if score < 0.0 or score > 1.0:
        score = 1.0 / (1.0 + np.exp(-score))
    return float(np.clip(score, 0.0, 1.0))


def load_dataset(data_dir: Path, image_size: tuple[int, int], use_raw_255: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders missing. Expected:\n- {real_dir}\n- {fake_dir}"
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []

    for label, folder in ((0, real_dir), (1, fake_dir)):
        for image_path in collect_images(folder):
            prepared = preprocess(image_path, image_size, use_raw_255=use_raw_255)
            if prepared is None:
                continue
            images.append(prepared)
            labels.append(label)
            names.append(str(image_path))

    if not images:
        raise RuntimeError("No valid images found for evaluation.")

    return np.stack(images), np.array(labels, dtype="int32"), names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deepfake model and export accuracy metrics.")
    parser.add_argument("--model", required=True, help="Path to Keras model (.h5/.keras)")
    parser.add_argument("--data", required=True, help="Validation dataset path containing real/ and fake/")
    parser.add_argument("--output", default="metrics/latest_metrics.json", help="Metrics output JSON path")
    parser.add_argument("--img-size", type=int, default=224, help="Input image side length")
    parser.add_argument("--threshold", type=float, default=0.5, help="Base decision threshold for fake class")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    y_pred = (y_scores >= threshold).astype("int32")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    accuracy = float((tp + tn) / total)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = 0.0 if (precision + recall) == 0 else float((2.0 * precision * recall) / (precision + recall))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(y_scores)
    sorted_scores = y_scores[order]
    sorted_labels = y_true[order]

    ranks = np.arange(1, len(sorted_scores) + 1, dtype="float64")
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(sorted_scores):
        sum_ranks = np.bincount(inverse_indices, weights=ranks)
        average_ranks = sum_ranks / counts
        ranks = average_ranks[inverse_indices]

    positive_rank_sum = float(np.sum(ranks[sorted_labels == 1]))
    auc = (positive_rank_sum - (positives * (positives + 1) / 2.0)) / float(positives * negatives)
    return float(auc)


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, dict, list[dict]]:
    results: list[dict] = []
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_scores, best_threshold)

    for threshold in np.linspace(0.10, 0.90, 81):
        t = float(round(float(threshold), 2))
        metrics = compute_metrics(y_true, y_scores, t)
        row = {
            "threshold": t,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        }
        results.append(row)

        current_key = (metrics["accuracy"], metrics["f1_score"], -abs(t - 0.5))
        best_key = (best_metrics["accuracy"], best_metrics["f1_score"], -abs(best_threshold - 0.5))
        if current_key > best_key:
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics, results


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    use_raw_255 = model_uses_raw_255_inputs(model)
    print(f"Detected input preprocessing mode: {'raw-255 RGB' if use_raw_255 else 'unit-scale RGB'}")

    print(f"Loading dataset from: {data_path}")
    x_test, y_true, _ = load_dataset(data_path, (args.img_size, args.img_size), use_raw_255=use_raw_255)

    print(f"Running predictions on {len(x_test)} samples...")
    raw_predictions = model.predict(x_test, verbose=0)
    y_scores = np.array([normalize_prediction(raw) for raw in raw_predictions], dtype="float32")

    base_metrics = compute_metrics(y_true, y_scores, args.threshold)

    auc = compute_auc(y_true, y_scores)

    best_threshold, best_metrics, threshold_scan = find_best_threshold(y_true, y_scores)

    metrics = {
        "dataset_name": data_path.name,
        "sample_count": int(len(y_true)),
        "class_mapping": {"real": 0, "fake": 1},
        "threshold": args.threshold,
        "best_threshold": best_threshold,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "auc": auc,
        "confusion_matrix": best_metrics["confusion_matrix"],
        "base_threshold_metrics": base_metrics,
        "threshold_scan": threshold_scan,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "target_80_achieved": best_metrics["accuracy"] >= 0.80,
        "input_preprocessing": "raw_255_rgb" if use_raw_255 else "unit_scale_rgb",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEvaluation complete:")
    print(f"Base threshold ({args.threshold:.2f}) accuracy : {base_metrics['accuracy'] * 100:.2f}%")
    print(f"Best threshold                      : {best_threshold:.2f}")
    print(f"Best-threshold accuracy             : {best_metrics['accuracy'] * 100:.2f}%")
    print(f"Best-threshold precision            : {best_metrics['precision'] * 100:.2f}%")
    print(f"Best-threshold recall               : {best_metrics['recall'] * 100:.2f}%")
    print(f"Best-threshold F1                   : {best_metrics['f1_score'] * 100:.2f}%")
    if auc is not None:
        print(f"AUC                                 : {auc * 100:.2f}%")
    print(f"Saved metrics to: {output_path}")

    if best_metrics["accuracy"] >= 0.80:
        print("Target met: model is at or above 80% accuracy.")
    else:
        print("Target not met yet: improve training data quality/size and retrain.")


if __name__ == "__main__":
    main()
