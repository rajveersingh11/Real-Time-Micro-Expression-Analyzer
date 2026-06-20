# Deployment and Running Guide

This document describes how to install dependencies, run the analyzer, launch the API, and deploy the application in production environments.

---

## 🛠️ Local Installation

### Prerequisites
- Python >= 3.10
- Git

### Steps
1. Clone the repository and navigate into the workspace.
2. Initialize virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate.ps1
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

---

## 🚀 Running the Services

### 1. Web Dashboard (Streamlit UI)
To launch the interactive dashboard for camera analysis and log viewing:
```bash
streamlit run src/inference/dashboard_app.py
```

### 2. FastAPI Server
To launch the REST API server:
```bash
mea-server
```
The server will start on `http://localhost:8000`.

### 3. CLI Analyzer
To run the OpenCV based command-line video/webcam analyzer:
```bash
mea-analyze --camera-index 0
```
Use `--record-landmarks` to save session files, or `--no-display` to run in headless mode.

---

## 🐳 Docker Deployment

To build and run the services inside a Docker container:

### 1. Build the Docker Image
```bash
docker build -t micro-expression-analyzer .
```

### 2. Run with Docker Compose
To run using Docker Compose which mounts persistent directories for models and logs:
```bash
docker-compose up --build
```
This binds to port `8000` on the host, allowing access to `/docs` and the WebSocket endpoint.
