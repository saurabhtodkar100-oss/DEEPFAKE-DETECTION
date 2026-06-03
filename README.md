# Deepfake Detection Portal (Image + Video)

A Flask web project that detects deepfake images/videos and generates analysis output for engineering project submission.

## Features

- Upload image/video from browser.
- Deepfake verdict (`REAL` / `FAKE`) with confidence score.
- Video frame sampling + fake-probability trend chart.
- JSON report output for each analysis (`reports/*.json`).
- Utilities to train and evaluate model.
- Automatic use of best validation threshold (from evaluation) during website inference.

## Project Folder

`D:\deepfake-detection-portal`

## Dataset Format (for training/evaluation)

```text
dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
```

Use balanced and clean data (e.g., FaceForensics++, Celeb-DF, DFDC subsets) to reach reliable 80%+ accuracy.

## Real Dataset Workflow

For better real-world performance, train on a larger dataset prepared from authentic and fake images/videos.

1. Put your raw media into separate folders, for example:

```text
D:/datasets/
  real_images/
  ai_images/
  real_videos/
  fake_videos/
```

2. Copy and edit [configs/real_training_sources.example.json](D:/deepfake-detection-portal/configs/real_training_sources.example.json) so it points at your dataset folders.

3. Build a training dataset with extracted video frames:

```powershell
python tools/prepare_real_dataset.py --config configs/real_training_sources.example.json --output-root dataset_large --frames-per-video 12 --face-crop-videos
```

This creates:

```text
dataset_large/
  train/
    real/
    fake/
  val/
    real/
    fake/
  dataset_manifest.json
```

The script hard-links images by default and extracts evenly spaced video frames into the same `train/val` class folders.

## Setup

```powershell
cd D:\deepfake-detection-portal
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Model

```powershell
python tools/train_model.py --train-dir dataset/train --val-dir dataset/val --output-model models/deepfake_model.h5
```

Notes:
- Training now forces class mapping: `real=0`, `fake=1`.
- Class weights are applied automatically for imbalanced datasets.

For a real dataset, use a stronger preset:

```powershell
python tools/train_model.py --train-dir dataset_large/train --val-dir dataset_large/val --output-model models/deepfake_model.keras --backbone efficientnetb3 --img-size 300 --batch-size 16 --epochs 10 --fine-tune-epochs 8 --crop-to-aspect-ratio
```

Options:
- `--backbone efficientnetb0|efficientnetb2|efficientnetb3|efficientnetb4`
- `--mixed-precision` if you have a supported GPU
- `--fine-tune-layers 64` to unfreeze more of the backbone on larger datasets

## Evaluate Accuracy and Tune Threshold

```powershell
python tools/evaluate_model.py --model models/deepfake_model.h5 --data dataset/val --output metrics/latest_metrics.json
```

For the real dataset workflow:

```powershell
python tools/evaluate_model.py --model models/deepfake_model.keras --data dataset_large/val --output metrics/latest_metrics.json
```

This computes base metrics, scans thresholds from `0.10` to `0.90`, and saves `best_threshold`.
The web app automatically uses this best threshold for later predictions.

## Run Website

```powershell
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Use Existing Model

If you already have a model file (`D:\deepfake_model.keras` or `D:\deepfake_model.h5`), the app can load it automatically.

## Output Files

- Analysis JSON reports: `reports/<report_id>.json`
- Uploaded files: `static/uploads/`
- Accuracy + threshold metrics: `metrics/latest_metrics.json`

## Important Note

If accuracy is below 80%, improve dataset quality, remove duplicate/noisy samples, keep class balance, and retrain.
The bundled sample dataset is only for smoke testing and will not produce strong real-world accuracy on AI-generated portraits or deepfake videos.
