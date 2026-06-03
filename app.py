from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

if __name__ == "__main__" and VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from analysis_engine import DeepfakeAnalyzer, load_metrics


UPLOAD_DIR = BASE_DIR / "static" / "uploads"
REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = BASE_DIR / "metrics" / "latest_metrics.json"


def resolve_runtime_threshold(default: float = 0.5) -> float:
    if not METRICS_PATH.exists():
        return default

    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default

    candidate = payload.get("best_threshold", payload.get("threshold", default))
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return default

    if value < 0.0 or value > 1.0:
        return default
    return value


def build_bootstrap_command() -> str:
    python_path = Path(sys.executable).resolve()
    script_path = (BASE_DIR / "tools" / "bootstrap_demo_model.py").resolve()
    return f'"{python_path}" "{script_path}"'


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "deepfake-portal-secret")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

for folder in (UPLOAD_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)


analyzer = DeepfakeAnalyzer(
    model_candidates=[
        os.getenv("DEEPFAKE_MODEL_PATH", ""),
        BASE_DIR / "models" / "deepfake_model.keras",
        BASE_DIR / "models" / "deepfake_model.h5",
        Path("D:/deepfake_model.keras"),
        Path("D:/deepfake_model.h5"),
    ],
    report_dir=REPORT_DIR,
    threshold=resolve_runtime_threshold(),
)


@app.template_filter("pct")
def pct(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_state=analyzer.model_state,
        metrics=load_metrics(METRICS_PATH),
        setup_command=build_bootstrap_command(),
    )


@app.post("/analyze")
def analyze():
    if not analyzer.model_state.ready:
        flash("Model is not ready. Generate the demo model or set DEEPFAKE_MODEL_PATH first.", "error")
        return redirect(url_for("index"))

    media = request.files.get("media")
    if media is None or not media.filename:
        flash("Please choose an image or video file.", "error")
        return redirect(url_for("index"))

    original_name = media.filename
    secured = secure_filename(original_name)
    if not secured:
        flash("Invalid file name.", "error")
        return redirect(url_for("index"))

    extension = Path(secured).suffix.lower()
    stored_name = f"{uuid4().hex[:10]}_{Path(secured).stem}{extension}"
    stored_path = UPLOAD_DIR / stored_name

    if not analyzer.is_supported(stored_path):
        flash("Unsupported file type. Use image or video formats only.", "error")
        return redirect(url_for("index"))

    media.save(stored_path)

    try:
        report = analyzer.analyze_file(stored_path, original_name=original_name)
    except Exception as exc:  # pylint: disable=broad-except
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("index"))

    return redirect(url_for("result", report_id=report["report_id"]))


@app.get("/result/<report_id>")
def result(report_id: str):
    report = analyzer.read_report(report_id)
    if not report:
        flash("Analysis report not found.", "error")
        return redirect(url_for("index"))

    frame_scores = report.get("frame_scores") or []
    frame_labels = [int(item["frame"]) for item in frame_scores]
    frame_values = [round(float(item["fake_probability"]) * 100, 2) for item in frame_scores]

    return render_template(
        "result.html",
        report=report,
        frame_labels=frame_labels,
        frame_values=frame_values,
    )


@app.get("/reports/<report_id>.json")
def download_report(report_id: str):
    report_file = analyzer.report_file_path(report_id)
    if not report_file.exists():
        flash("Report file not found.", "error")
        return redirect(url_for("index"))

    return send_file(report_file, as_attachment=True, download_name=f"deepfake_report_{report_id}.json")


@app.errorhandler(413)
def handle_large_upload(_):
    flash("File is too large. Maximum allowed size is 200 MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
