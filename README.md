# 🔬 Real-Time Micro-Expression & Stress Analyzer

[![Tests](https://github.com/rajveersingh11/Real-Time-Micro-Expression-Analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/rajveersingh11/Real-Time-Micro-Expression-Analyzer/actions/workflows/test.yml)
[![Docker](https://img.shields.io/badge/docker-container-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

A production-grade, real-time computer vision system for facial micro-expression analysis and psychological stress estimation. Built using PyTorch, MediaPipe Face Landmarker, FastAPI (REST & WebSockets), and Streamlit.

---

## ✨ Features

- **📹 Live Web Dashboard (Streamlit)**: Streamlit-based UI for real-time webcam analysis, metrics plotting, image upload profiling, and logged session file statistics.
- **⚡ Real-Time API (REST & WebSockets)**: Dual API layer offering stateless frame analysis (`POST /analyze`) and stateful WebSocket streaming (`WS /ws/stream`) with connection-specific signal history.
- **📈 Advanced Feature Engineering**: Calculates 5 key micro-expression indicators (eyebrow raise, lip tension, EAR blink intensity, head nod, cheek asymmetry delta).
- **🌀 Robust Temporal Filtering**: Exponential Moving Average (EMA) signal smoothing with automated out-of-bounds parameter validation.
- **🧠 Custom Training Factory**: Supports multiple model backbones (ResNet18, EfficientNet B0, MobileNet V3), Adam/AdamW/SGD optimizers, cosine/step learning rate schedules, and label smoothing.
- **🐳 Dockerized Deployment**: Easy multi-platform setup with preinstalled OpenCV dependencies inside docker containers.
- **🗄️ Local SQLite Session Database**: Stores session metadata, user profiles, average/peak stress levels, and references to logging outputs without runtime overhead.
- **🛡️ Secure & Optimized Performance**: Secured serialization patterns (`weights_only=True`), protected against memory leaks and Windows multiprocessing issues, and includes automated testing (pytest).

---

## 🏗️ Folder Structure

```
Real-Time-Micro-Expression-Analyzer/
├── config/
│   ├── default.yaml            # Primary model & training configuration
│   └── config.yaml             # Dataset download configuration
├── docs/                       # Project documentation
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── CONTRIBUTING.md
├── src/
│   ├── api/                    # FastAPI app and endpoint routers
│   ├── inference/              # Landmark detection, features, and dashboard
│   ├── models/                 # CNN architectures and model factories
│   ├── preprocessing/          # Image loading and label alignment
│   ├── training/               # Losses, schedulers, and epoch trainers
│   └── utils/                  # Privacy, config loaders, exceptions, and logging
├── tests/                      # Pytest unit and integration test suite
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── pyproject.toml              # Editable script entry points
└── requirements.txt            # Pinned requirements
```

---

## 🚀 Quickstart

### 1. Environment Setup

Since this project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management, you can sync the environment and install all packages in editable mode with a single command:

```bash
# Sync dependencies and set up virtual environment (.venv)
uv sync
```

To run package scripts, prefix them with `uv run`, or activate the environment:
```bash
# Activating virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 2. Dataset Configuration and Downloading

Dataset download configurations (including output directories and Kaggle IDs) are managed in [config/config.yaml](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/config/config.yaml).

Ensure you have your Kaggle credentials file (`kaggle.json`) placed under `~/.kaggle/` (Linux/macOS) or `C:\Users\<User>\.kaggle\` (Windows). Then download and unzip the datasets automatically:

```bash
mea-download
```

To run dataset preprocessing and create 70/15/15 train/val/test splits:
```bash
python -m src.preprocessing.dataset_unifier
```

---

## 🏃 Running the Application

### 🖥️ 1. Streamlit Web Dashboard
Launches the browser-based visualization dashboard:
```bash
streamlit run src/inference/dashboard_app.py
```

### 🌐 2. FastAPI API Server
Launches the REST and WebSocket streaming server:
```bash
mea-server
```
Interactive documentation is available at `http://localhost:8000/docs`.

### 💻 3. Command-Line Analyzer
Launches the OpenCV webcam analyzer directly in your terminal:
```bash
# Basic running
mea-analyze --camera-index 0

# Running with landmark sequence recording enabled
mea-analyze --camera-index 0 --record-landmarks
```

### 🏋️ 4. ML Model Training
Launches model training using configurations defined in `config/default.yaml`:
```bash
python -m src.training.train
```

---

## 🧪 Testing

We use `pytest` for unit and integration testing. Run the following command:
```bash
pytest
```

---

## 🐳 Docker Setup

To run the complete FastAPI application in containerized mode:
```bash
docker-compose up --build
```
The server will bind to `http://localhost:8000` with volume mounts for persistent logs and checkpoints.

---

## 📖 Documentation Index

- **[ARCHITECTURE.md](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/docs/ARCHITECTURE.md)**: Details landmark extraction mathematical formulations, signal smoothing, and model architectures.
- **[API.md](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/docs/API.md)**: Lists FastAPI path parameters, query structures, base64 payloads, and WS streaming responses.
- **[DEPLOYMENT.md](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/docs/DEPLOYMENT.md)**: Extended setup configurations for local and Dockerized operations.
- **[CONTRIBUTING.md](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/docs/CONTRIBUTING.md)**: Coding conventions, writing test assertions, and pytest guidelines.
