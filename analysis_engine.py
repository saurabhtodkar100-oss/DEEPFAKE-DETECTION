from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def model_uses_raw_255_inputs(model: Any) -> bool:
    """Detect models that already normalize internally via Rescaling layers."""
    visited: set[int] = set()

    def contains_rescaling(layer: Any) -> bool:
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


def verdict_from_probability(fake_probability: float, threshold: float = 0.5) -> str:
    return "FAKE" if fake_probability >= threshold else "REAL"


def confidence_from_probability(fake_probability: float, verdict: str) -> float:
    if verdict == "FAKE":
        return float(np.clip(fake_probability, 0.0, 1.0))
    return float(np.clip(1.0 - fake_probability, 0.0, 1.0))


def risk_level_from_probability(fake_probability: float) -> str:
    if fake_probability >= 0.8:
        return "High"
    if fake_probability >= 0.6:
        return "Medium"
    return "Low"


def _as_percentage(value: float | None) -> str:
    if value is None:
        return "Not available"
    return f"{value * 100:.2f}%"


def _skin_ratio(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float(np.mean(skin_mask > 0))


def _edge_density(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean() / 255.0)


def _noise_residual(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(np.mean(np.abs(gray - blurred)))


def _local_texture_std(frame: np.ndarray, grid_size: int = 8) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = np.array_split(gray, grid_size, axis=0)
    blocks = [block for row in rows for block in np.array_split(row, grid_size, axis=1)]
    if not blocks:
        return 0.0
    return float(np.mean([np.std(block) for block in blocks]))


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    default_metrics = {
        "dataset_name": "Not evaluated yet",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "auc": None,
        "threshold": 0.5,
        "best_threshold": 0.5,
        "evaluated_at": None,
        "status": "Pending evaluation",
        "notes": "Run tools/evaluate_model.py after training to populate real metrics.",
    }

    if not metrics_path.exists():
        return default_metrics

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_metrics

    metrics = {**default_metrics, **payload}
    accuracy = metrics.get("accuracy")
    if isinstance(accuracy, (int, float)):
        metrics["status"] = "Target met (>=80%)" if accuracy >= 0.80 else "Below target (<80%)"
    return metrics


@dataclass
class ModelState:
    ready: bool
    path: str | None
    error: str | None = None


class DeepfakeAnalyzer:
    def __init__(
        self,
        model_candidates: Iterable[os.PathLike[str] | str],
        report_dir: Path,
        image_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
    ) -> None:
        self.model_candidates = [Path(candidate) for candidate in model_candidates if candidate]
        self.report_dir = report_dir
        self.image_size = image_size
        self.threshold = threshold
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_frame_dir = self.report_dir.parent / "static" / "analysis_frames"
        self.analysis_frame_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._model_uses_raw_inputs = False
        self._model_state = ModelState(ready=False, path=None, error="Model is not loaded yet.")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_detector = cv2.CascadeClassifier(cascade_path)
        self._has_face_detector = not self._face_detector.empty()

    @property
    def model_state(self) -> ModelState:
        self._ensure_model_loaded()
        return self._model_state

    def is_supported(self, file_path: Path) -> bool:
        suffix = file_path.suffix.lower()
        return suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS

    def media_type(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

    def analyze_file(self, file_path: Path, original_name: str | None = None) -> dict[str, Any]:
        self._ensure_model_loaded()
        if not self._model_state.ready:
            raise RuntimeError(self._model_state.error or "Model failed to load.")

        report_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        media_type = self.media_type(file_path)
        if media_type == "image":
            result = self._analyze_image(file_path)
        elif media_type == "video":
            result = self._analyze_video(file_path, report_id=report_id)
        else:
            raise ValueError("Unsupported file type.")

        report = {
            "report_id": report_id,
            "filename": original_name or file_path.name,
            "stored_filename": file_path.name,
            "upload_relative_path": f"uploads/{file_path.name}",
            "media_type": media_type,
            "threshold": self.threshold,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        self._write_report(report_id, report)
        return report

    def read_report(self, report_id: str) -> dict[str, Any] | None:
        target = self.report_dir / f"{report_id}.json"
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def report_file_path(self, report_id: str) -> Path:
        return self.report_dir / f"{report_id}.json"

    def _write_report(self, report_id: str, payload: dict[str, Any]) -> None:
        target = self.report_dir / f"{report_id}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._model_state.ready:
            return

        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            self._model_state = ModelState(
                ready=False,
                path=None,
                error=f"TensorFlow is unavailable: {exc}",
            )
            return

        existing_model = next((path for path in self.model_candidates if path.exists()), None)
        if not existing_model:
            searched = ", ".join(str(path) for path in self.model_candidates) or "No paths configured"
            self._model_state = ModelState(
                ready=False,
                path=None,
                error=f"Model file not found. Checked: {searched}",
            )
            return

        try:
            self._model = load_model(existing_model)
        except Exception as exc:  # pylint: disable=broad-except
            self._model_state = ModelState(
                ready=False,
                path=str(existing_model),
                error=f"Model failed to load: {exc}",
            )
            return

        self._model_uses_raw_inputs = model_uses_raw_255_inputs(self._model)
        self._model_state = ModelState(ready=True, path=str(existing_model), error=None)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.image_size)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        prepared = resized.astype("float32")
        if not self._model_uses_raw_inputs:
            prepared = prepared / 255.0
        return np.expand_dims(prepared, axis=0)

    def _normalize_prediction(self, prediction: Any) -> float:
        values = np.array(prediction).astype("float32").squeeze()
        if values.ndim == 0:
            score = float(values)
        elif values.size == 1:
            score = float(values.reshape(-1)[0])
        elif values.size == 2:
            max_value = np.max(values)
            exp_scores = np.exp(values - max_value)
            probabilities = exp_scores / np.sum(exp_scores)
            score = float(probabilities.reshape(-1)[1])
        else:
            score = float(np.mean(values))

        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + np.exp(-score))
        return float(np.clip(score, 0.0, 1.0))

    def _predict_probability(self, frame: np.ndarray, use_tta: bool = False) -> float:
        prepared = self._preprocess_frame(frame)
        raw = self._model.predict(prepared, verbose=0)
        scores = [self._normalize_prediction(raw)]

        if use_tta:
            flipped = cv2.flip(frame, 1)
            prepared_flipped = self._preprocess_frame(flipped)
            raw_flipped = self._model.predict(prepared_flipped, verbose=0)
            scores.append(self._normalize_prediction(raw_flipped))

        return float(np.mean(scores))

    def _thumbnail_bytes(self, frame: np.ndarray, max_width: int = 360) -> bytes | None:
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            resized = cv2.resize(frame, (max_width, max(1, int(height * scale))))
        else:
            resized = frame

        success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            return None
        return encoded.tobytes()

    def _build_key_frames(
        self,
        report_id: str,
        frame_scores: list[dict[str, float | int | None]],
        thumbnails: dict[int, bytes],
        max_per_group: int = 5,
    ) -> list[dict[str, Any]]:
        target_dir = self.analysis_frame_dir / report_id
        target_dir.mkdir(parents=True, exist_ok=True)

        sorted_desc = sorted(frame_scores, key=lambda item: float(item["fake_probability"]), reverse=True)
        sorted_asc = sorted(frame_scores, key=lambda item: float(item["fake_probability"]))

        combined: list[tuple[dict[str, float | int | None], str]] = []
        selected_ids: set[int] = set()

        for item in sorted_desc:
            frame_id = int(item["frame"])
            if frame_id in selected_ids:
                continue
            combined.append((item, "Most Fake"))
            selected_ids.add(frame_id)
            if len([1 for _, label in combined if label == "Most Fake"]) >= max_per_group:
                break

        real_count = 0
        for item in sorted_asc:
            frame_id = int(item["frame"])
            if frame_id in selected_ids:
                continue
            combined.append((item, "Most Real"))
            selected_ids.add(frame_id)
            real_count += 1
            if real_count >= max_per_group:
                break

        key_frames: list[dict[str, Any]] = []
        for rank, (item, label) in enumerate(combined, start=1):
            frame_id = int(item["frame"])
            image_bytes = thumbnails.get(frame_id)
            if not image_bytes:
                continue

            suffix = "fake" if label == "Most Fake" else "real"
            file_name = f"{rank:02d}_{suffix}_f{frame_id}.jpg"
            file_path = target_dir / file_name
            file_path.write_bytes(image_bytes)

            key_frames.append(
                {
                    "frame": frame_id,
                    "label": label,
                    "fake_probability": float(item["fake_probability"]),
                    "background_probability": float(item["background_probability"]),
                    "face_probability": None if item["face_probability"] is None else float(item["face_probability"]),
                    "thumbnail_relative_path": f"analysis_frames/{report_id}/{file_name}",
                }
            )

        return key_frames

    def _blend_face_background(self, face_probability: float | None, background_probability: float) -> tuple[float, str]:
        if face_probability is None:
            return background_probability, "background-only"

        blended = float(np.clip((0.85 * face_probability) + (0.15 * background_probability), 0.0, 1.0))
        note = "face-priority blend (85:15)"

        if abs(face_probability - background_probability) >= 0.35:
            blended = float(np.clip((0.93 * face_probability) + (0.07 * background_probability), 0.0, 1.0))
            note = "face-priority blend (93:7 due to high disagreement)"

        return blended, note

    def _iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        inter = float((ix1 - ix0) * (iy1 - iy0))
        area_a = float((ax1 - ax0) * (ay1 - ay0))
        area_b = float((bx1 - bx0) * (by1 - by0))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _detect_faces(self, frame: np.ndarray, max_faces: int = 2, min_area_ratio: float = 0.0) -> list[tuple[int, int, int, int, int]]:
        if not self._has_face_detector:
            return []

        height, width = frame.shape[:2]
        min_side = max(60, min(height, width) // 40)
        min_side = min(min_side, 120)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )

        candidates: list[tuple[int, int, int, int, int]] = []
        for x, y, w, h in detections:
            pad_x = int(0.15 * w)
            pad_y = int(0.15 * h)

            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(width, x + w + pad_x)
            y1 = min(height, y + h + pad_y)

            box_w = x1 - x0
            box_h = y1 - y0
            if box_w < min_side or box_h < min_side:
                continue

            area = box_w * box_h
            if min_area_ratio > 0 and area < int(min_area_ratio * height * width):
                continue
            candidates.append((x0, y0, x1, y1, area))

        candidates.sort(key=lambda item: item[4], reverse=True)

        selected: list[tuple[int, int, int, int, int]] = []
        for candidate in candidates:
            c_box = candidate[:4]
            if any(self._iou(c_box, chosen[:4]) > 0.45 for chosen in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_faces:
                break

        return selected

    def _weighted_average(self, values: list[float], weights: list[float]) -> float:
        if not values:
            return 0.0
        if not weights or len(values) != len(weights):
            return float(np.mean(values))

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return float(np.mean(values))

        return float(np.sum(np.array(values, dtype="float32") * np.array(weights, dtype="float32")) / weight_sum)

    def _is_likely_human_face(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False

        height, width = crop.shape[:2]
        if min(height, width) < 48:
            return False

        skin = _skin_ratio(crop)
        if skin < 0.08:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean() / 255.0)
        value = float(hsv[:, :, 2].mean() / 255.0)
        return saturation >= 0.08 and value >= 0.12

    def _image_heuristic_adjustment(
        self,
        frame: np.ndarray,
        face_count: int,
        rejected_face_regions: int,
        max_face_area_ratio: float,
    ) -> tuple[float, list[str]]:
        height, width = frame.shape[:2]
        min_dimension = min(height, width)
        max_dimension = max(height, width)
        edge_density = _edge_density(frame)
        noise_residual = _noise_residual(frame)
        local_texture = _local_texture_std(frame)

        adjustment = 0.0
        notes: list[str] = []

        if face_count >= 1 and min_dimension <= 384:
            adjustment += 0.03
            notes.append("Small face-centric web image increased fake probability.")
        elif face_count >= 1 and min_dimension <= 512:
            adjustment += 0.02
            notes.append("Low-resolution face image increased fake probability slightly.")

        if (
            face_count >= 1
            and max_face_area_ratio >= 0.45
            and min_dimension <= 1024
            and edge_density <= 0.018
            and noise_residual <= 1.0
        ):
            adjustment += 0.03
            notes.append("Large smooth portrait face increased fake probability.")

        if min_dimension >= 1000 and max_dimension >= 2500 and edge_density < 0.015 and noise_residual >= 0.75:
            adjustment -= 0.025
            notes.append("Large camera-style photo reduced fake probability.")
        elif min_dimension >= 900 and edge_density < 0.012 and noise_residual >= 0.75:
            adjustment -= 0.015
            notes.append("Mid-size camera-style photo reduced fake probability slightly.")

        if (
            face_count >= 1
            and min_dimension >= 1800
            and max_face_area_ratio <= 0.16
            and noise_residual <= 1.5
        ):
            adjustment -= 0.025
            notes.append("High-resolution camera framing reduced fake probability.")

        if rejected_face_regions > 0:
            notes.append("Ignored face-like regions that did not resemble human skin.")

        if face_count == 0 and local_texture >= 55.0 and min_dimension >= 1200:
            adjustment -= 0.01
            notes.append("High local texture consistency favored a real-photo interpretation.")

        return float(np.clip(adjustment, -0.05, 0.05)), notes

    def _video_heuristic_adjustment(
        self,
        *,
        face_frame_ratio: float,
        avg_face_probability: float | None,
        avg_background_probability: float,
        motion_mean: float,
        motion_std: float,
        score_std: float,
        sampled_frames: int,
    ) -> tuple[float, float, list[str]]:
        adjustment = 0.0
        threshold_offset = 0.0
        notes: list[str] = []

        if (
            sampled_frames >= 24
            and face_frame_ratio <= 0.18
            and avg_background_probability >= (self.threshold - 0.02)
            and motion_mean <= 2.8
            and motion_std <= 1.2
            and score_std <= 0.001
        ):
            adjustment += 0.05
            threshold_offset -= 0.005
            notes.append("Low-motion video synthesis cues increased fake probability.")
        elif (
            sampled_frames >= 20
            and face_frame_ratio <= 0.20
            and avg_background_probability >= (self.threshold - 0.015)
            and motion_mean <= 3.6
            and motion_std <= 1.8
            and score_std <= 0.0025
        ):
            adjustment += 0.03
            notes.append("Stable frame-to-frame motion pattern increased fake probability slightly.")

        if face_frame_ratio >= 0.15 and avg_face_probability is not None and avg_face_probability >= self.threshold:
            adjustment += 0.015
            notes.append("Repeated face-region suspicion increased fake probability slightly.")

        if face_frame_ratio <= 0.10 and (motion_mean >= 7.0 or motion_std >= 4.0):
            adjustment -= 0.015
            threshold_offset += 0.005
            notes.append("High natural motion reduced fake probability slightly.")
        elif face_frame_ratio <= 0.15 and motion_mean >= 5.0 and motion_std >= 2.5:
            adjustment -= 0.01
            notes.append("Stronger natural motion cues favored a real-video interpretation.")

        if score_std <= 0.001 and motion_mean <= 2.2:
            notes.append("Very flat frame scores indicated weak temporal separation from the base model.")

        return float(np.clip(adjustment, -0.05, 0.06)), float(np.clip(threshold_offset, -0.02, 0.02)), notes

    def _face_probability(
        self,
        frame: np.ndarray,
        max_faces: int = 2,
        min_area_ratio: float = 0.0,
    ) -> tuple[float | None, int, int, float]:
        face_boxes = self._detect_faces(frame, max_faces=max_faces, min_area_ratio=min_area_ratio)
        if not face_boxes:
            return None, 0, 0, 0.0

        probs: list[float] = []
        areas: list[float] = []
        rejected_faces = 0
        height, width = frame.shape[:2]
        frame_area = float(max(height * width, 1))

        for x0, y0, x1, y1, area in face_boxes:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if not self._is_likely_human_face(crop):
                rejected_faces += 1
                continue
            probs.append(self._predict_probability(crop, use_tta=True))
            areas.append(float(area))

        if not probs:
            return None, 0, rejected_faces, 0.0

        max_area_ratio = max(areas) / frame_area if areas else 0.0
        return self._weighted_average(probs, areas), len(probs), rejected_faces, float(max_area_ratio)

    def _analyze_image(self, image_path: Path) -> dict[str, Any]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError("Failed to read the uploaded image.")

        whole_probability = self._predict_probability(frame, use_tta=True)
        face_probability, face_count, rejected_faces, max_face_area_ratio = self._face_probability(
            frame,
            max_faces=2,
            min_area_ratio=0.002,
        )
        base_probability, blend_note = self._blend_face_background(face_probability, whole_probability)
        heuristic_adjustment, heuristic_notes = self._image_heuristic_adjustment(
            frame,
            face_count=face_count,
            rejected_face_regions=rejected_faces,
            max_face_area_ratio=max_face_area_ratio,
        )
        fake_probability = float(np.clip(base_probability + heuristic_adjustment, 0.0, 1.0))

        verdict = verdict_from_probability(fake_probability, self.threshold)
        confidence = confidence_from_probability(fake_probability, verdict)
        risk_level = risk_level_from_probability(fake_probability)

        observations = [
            f"Whole-image fake probability: {_as_percentage(whole_probability)}.",
            f"Model blend probability before heuristics: {_as_percentage(base_probability)}.",
            f"Final calibrated fake probability: {_as_percentage(fake_probability)}.",
            f"Confidence in final verdict: {_as_percentage(confidence)}.",
            f"Risk level is {risk_level.lower()} at threshold {_as_percentage(self.threshold)}.",
        ]

        if face_probability is not None:
            observations.append(
                f"Face-focused probability: {_as_percentage(face_probability)} using {face_count} detected face region(s)."
            )
        else:
            observations.append("No clear face region detected, so full-image score was used.")

        observations.append(f"Decision strategy: {blend_note}.")
        if heuristic_adjustment != 0.0:
            direction = "increased" if heuristic_adjustment > 0 else "reduced"
            observations.append(
                f"Lightweight visual heuristics {direction} the fake probability by {_as_percentage(abs(heuristic_adjustment))}."
            )
        observations.extend(heuristic_notes)

        if verdict == "FAKE":
            observations.append("Potential manipulated patterns detected in facial texture and blending cues.")
        else:
            observations.append("No strong manipulation signal detected for this image.")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "model_blend_probability": base_probability,
            "whole_image_probability": whole_probability,
            "face_probability": face_probability,
            "face_regions_used": face_count,
            "max_face_area_ratio": max_face_area_ratio,
            "rejected_face_regions": rejected_faces,
            "heuristic_adjustment": heuristic_adjustment,
            "blend_note": blend_note,
            "risk_level": risk_level,
            "sampled_frames": 1,
            "total_frames": 1,
            "suspicious_ratio": float(fake_probability >= self.threshold),
            "frame_scores": [{"frame": 0, "fake_probability": fake_probability}],
            "top_suspicious_frames": [{"frame": 0, "fake_probability": fake_probability}],
            "observations": observations,
        }

    def _analyze_video(self, video_path: Path, report_id: str) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("Failed to open the uploaded video.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(total_frames // 80, 1) if total_frames else 5

        frame_scores: list[dict[str, float | int | None]] = []
        thumbnail_map: dict[int, bytes] = {}
        used_scores: list[float] = []
        background_scores: list[float] = []
        face_scores: list[float] = []
        motion_scores: list[float] = []

        face_frame_count = 0
        frame_index = 0
        previous_sample_gray: np.ndarray | None = None

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % stride == 0:
                background_probability = self._predict_probability(frame, use_tta=False)
                face_probability, _, _, _ = self._face_probability(frame, max_faces=2, min_area_ratio=0.004)
                frame_probability, _ = self._blend_face_background(face_probability, background_probability)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                used_scores.append(frame_probability)
                background_scores.append(background_probability)

                if face_probability is not None:
                    face_frame_count += 1
                    face_scores.append(face_probability)

                frame_motion = None
                if previous_sample_gray is not None:
                    frame_motion = float(
                        np.mean(np.abs(gray.astype("float32") - previous_sample_gray.astype("float32")))
                    )
                    motion_scores.append(frame_motion)
                previous_sample_gray = gray

                frame_scores.append(
                    {
                        "frame": frame_index,
                        "fake_probability": frame_probability,
                        "background_probability": background_probability,
                        "face_probability": face_probability,
                        "frame_motion": frame_motion,
                    }
                )

                thumb = self._thumbnail_bytes(frame)
                if thumb is not None:
                    thumbnail_map[frame_index] = thumb
            frame_index += 1

        capture.release()

        if not frame_scores:
            raise RuntimeError("No valid frames found in video.")

        score_values = np.array(used_scores, dtype="float32")
        avg_probability = float(np.mean(score_values))
        p75_probability = float(np.quantile(score_values, 0.75))
        score_std = float(np.std(score_values))
        suspicious_ratio = float(np.mean(score_values >= self.threshold))

        avg_background_probability = float(np.mean(np.array(background_scores, dtype="float32")))
        avg_face_probability = float(np.mean(np.array(face_scores, dtype="float32"))) if face_scores else None

        sampled_frames = len(frame_scores)
        face_frame_ratio = float(face_frame_count / sampled_frames) if sampled_frames else 0.0
        motion_mean = float(np.mean(np.array(motion_scores, dtype="float32"))) if motion_scores else 0.0
        motion_std = float(np.std(np.array(motion_scores, dtype="float32"))) if motion_scores else 0.0

        p75_lift = float(max(0.0, p75_probability - avg_probability))

        if face_frame_ratio >= 0.15:
            combined_score = float(np.clip(avg_probability + (0.30 * p75_lift) + (0.035 * suspicious_ratio), 0.0, 1.0))
            effective_threshold = self.threshold
            decision_strategy = "face-enabled temporal scoring"
        else:
            suspicious_activation = float(np.clip((avg_probability - (self.threshold - 0.01)) / 0.05, 0.0, 1.0))
            weighted_suspicious_ratio = suspicious_ratio * (0.40 + (0.60 * suspicious_activation))
            combined_score = float(
                np.clip(
                    avg_probability + (0.18 * p75_lift) + (0.025 * weighted_suspicious_ratio),
                    0.0,
                    1.0,
                )
            )
            effective_threshold = float(max(self.threshold + 0.02, 0.54))
            decision_strategy = "background-led calibrated scoring"

        heuristic_adjustment, threshold_offset, heuristic_notes = self._video_heuristic_adjustment(
            face_frame_ratio=face_frame_ratio,
            avg_face_probability=avg_face_probability,
            avg_background_probability=avg_background_probability,
            motion_mean=motion_mean,
            motion_std=motion_std,
            score_std=score_std,
            sampled_frames=sampled_frames,
        )
        combined_score = float(np.clip(combined_score + heuristic_adjustment, 0.0, 1.0))
        effective_threshold = float(np.clip(effective_threshold + threshold_offset, 0.45, 0.65))

        verdict = verdict_from_probability(combined_score, effective_threshold)
        confidence = confidence_from_probability(combined_score, verdict)
        risk_level = risk_level_from_probability(combined_score)

        top_frames = sorted(frame_scores, key=lambda item: float(item["fake_probability"]), reverse=True)[:8]
        top_frames = [
            {
                "frame": int(item["frame"]),
                "fake_probability": float(item["fake_probability"]),
                "background_probability": float(item["background_probability"]),
                "face_probability": None if item["face_probability"] is None else float(item["face_probability"]),
                "frame_motion": None if item["frame_motion"] is None else float(item["frame_motion"]),
            }
            for item in top_frames
        ]

        key_frames = self._build_key_frames(report_id, frame_scores, thumbnail_map, max_per_group=5)

        observations = [
            f"Average blended frame probability: {_as_percentage(avg_probability)}.",
            f"75th percentile frame probability: {_as_percentage(p75_probability)}.",
            f"Frame-score standard deviation: {_as_percentage(score_std)}.",
            f"Suspicious frame ratio: {_as_percentage(suspicious_ratio)}.",
            f"Average background probability: {_as_percentage(avg_background_probability)}.",
            f"Average face probability: {_as_percentage(avg_face_probability)}.",
            f"Face regions detected in {_as_percentage(face_frame_ratio)} of sampled frames.",
            f"Average sampled-frame motion: {motion_mean:.2f}.",
            f"Frame-motion standard deviation: {motion_std:.2f}.",
            f"Decision strategy: {decision_strategy}.",
            f"Effective video threshold: {_as_percentage(effective_threshold)}.",
            f"Final video score: {_as_percentage(combined_score)} with confidence {_as_percentage(confidence)}.",
            f"Generated {len(key_frames)} key-frame analysis images.",
        ]
        if heuristic_adjustment != 0.0:
            direction = "increased" if heuristic_adjustment > 0 else "reduced"
            observations.append(
                f"Video heuristics {direction} the fake probability by {_as_percentage(abs(heuristic_adjustment))}."
            )
        observations.extend(heuristic_notes)

        if verdict == "FAKE":
            observations.append("A sustained manipulation signal was detected across video frames.")
        else:
            observations.append("Video appears consistent with authentic content under current model settings.")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": combined_score,
            "average_frame_probability": avg_probability,
            "average_background_probability": avg_background_probability,
            "average_face_probability": avg_face_probability,
            "frame_score_std": score_std,
            "average_frame_motion": motion_mean,
            "frame_motion_std": motion_std,
            "face_frames": face_frame_count,
            "face_frame_ratio": face_frame_ratio,
            "effective_video_threshold": effective_threshold,
            "decision_strategy": decision_strategy,
            "heuristic_adjustment": heuristic_adjustment,
            "risk_level": risk_level,
            "sampled_frames": sampled_frames,
            "total_frames": total_frames if total_frames > 0 else frame_index,
            "suspicious_ratio": suspicious_ratio,
            "frame_scores": frame_scores,
            "top_suspicious_frames": top_frames,
            "key_frames": key_frames,
            "observations": observations,
        }
