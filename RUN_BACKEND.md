# AI-MicroExpression-Analyzer: Execution Guide

This document explains how to set up and run the real-time analyzer, offline analytics, and evaluation systems.

---

## 🛠️ 1. Environment Setup

### Prerequisites
- Python 3.10+ (Recommended)
- Webcam (Internal or External)

### Dependency Installation
Ensure all required libraries are installed:
```bash
pip install -r requirements.txt
```

### Model Initialization
The system uses a 5.6MB MediaPipe `face_landmarker.task` model. On the first run, the system will **automatically download** this file into the `models/` directory.

---

## 🎥 2. Real-Time Stress Detection

The primary application is `realtime_analyzer.py`. This script starts a webcam feed, detects landmarks, and calculates stress levels.

### Standard UI Mode
```bash
python src/inference/realtime_analyzer.py
```

### Advanced CLI Options
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--camera-index` | Select camera device ID (0, 1, etc.) | `0` |
| `--no-display` | Headless mode (no UI window) | `False` |
| `--verbose` | Print frame metrics in the terminal | `False` |
| `--alpha` | Smoothing factor (0.1 to 1.0) | `0.3` |
| `--record-landmarks` | Capture landmark sequences to `.npy` | `False` |
| `--log-path` | Custom directory for CSV logs | `logs` |

### Use Case Examples
- **Debugging**: `python src/inference/realtime_analyzer.py --verbose`
- **Headless Data Gathering**: `python src/inference/realtime_analyzer.py --no-display --record-landmarks`
- **Extra Smooth Analysis**: `python src/inference/realtime_analyzer.py --alpha 0.1`

---

## 📊 3. Offline Session Analysis

To analyze a previously recorded landmark session (`.npy` file), use the `session_analyzer` tool. This will generate temporal trend graphs for stress and facial behavior.

```bash
python src/analysis/session_analyzer.py data/processed/landmark_sessions/session_XYZ.npy
```

---

## ✅ 4. Model Evaluation

If you are developing or fine-tuning the underlying emotion classification model, run the evaluation suite to generate a performance report.

```bash
python src/evaluation/evaluator.py
```
**Outputs**:
- `logs/metrics.json`: Accuracy/F1 scores.
- `logs/confusion_matrix.png`: Prediction heatmap.
- `logs/classification_report.txt`: Per-emotion precision/recall.

---

## ⌨️ 5. Controls
- **'q'**: Quit the active webcam window.
- **Ctrl + C**: Terminate the process in the terminal.

## 📁 6. Data Storage
- **CSV Logs**: Stored in `logs/` (timestamped per session).
- **Landmark Data**: Stored in `data/processed/landmark_sessions/` (as NumPy binaries).
- **Visual Artifacts**: Stored in `logs/` (images and reports).
