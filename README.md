# AI-MicroExpression-Analyzer

## Project Overview
This project is dedicated to professional machine learning workflows for analyzing micro-expressions. It supports:
- Emotion model training
- MediaPipe landmark extraction
- Real-time stress inference
- Automatic dataset download and preprocessing
- Experiment logging

## Environment Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd AI-MicroExpression-Analyzer
```

### 2. Setup virtual environment and install dependencies

#### On Linux/macOS:
```bash
bash setup_env.sh
```

#### On Windows (PowerShell):
```powershell
.\setup_env.ps1
```

## Dataset Download
The project uses the Kaggle API to automatically download required datasets (`FER2013`, `RAF-DB`, `CK+`).

### 1. Kaggle API Configuration
Ensure you have your `kaggle.json` API key in `~/.kaggle/` (Linux/macOS) or `C:\Users\<User>\.kaggle\` (Windows).

### 2. Run the download script
```bash
# After activating your virtual environment
python download_data.py
```

This will:
- Create `data/raw/` directory.
- Download and unzip datasets automatically.
- Prepare the folder structure for preprocessing.
