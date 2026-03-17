# AI-MicroExpression-Analyzer: Backend Architecture

This document provides a detailed overview of the backend systems, data processing pipelines, and algorithmic logic used in the AI-MicroExpression-Analyzer.

## 🏗️ Architectural Overview

The backend is built using a modular Python architecture designed for high-frequency real-time inference. The system follows a linear data transformation pipeline:

**Pixels ➔ Landmarks ➔ Features ➔ Smoothed Signals ➔ Stress Score ➔ Classification**

---

## 🧩 Core Components

### 1. Facial Landmark Detection (`FaceLandmarkDetector`)
- **Engine**: MediaPipe Face Landmarker (Tasks API).
- **Output**: 478 normalized 3D coordinates (X, Y, Z).
- **Capability**: Includes high-fidelity iris tracking and facial contours.
- **Optimization**: Uses the `VIDEO` running mode for temporal consistency across frames.

### 2. Feature Engineering (`FeatureEngineer`)
This module translates raw geometric points into interpretable behavioral signals:
- **Eyebrow Raise**: Vertical distance between eyebrows and upper eyelids relative to face height.
- **Lip Tension**: The width-to-height ratio of the mouth, detecting compression.
- **Blink Intensity**: Eye Aspect Ratio (EAR). Lower values indicate closed eyes or high-frequency blinking.
- **Head Nod**: Delta Y movement of the nose tip between consecutive frames.
- **Facial Symmetry**: Comparison of Euclidean distances from the nose tip to the left and right cheeks.

### 3. Signal Smoothing (`SignalSmoother`)
To eliminate camera noise and micro-tremors, all features pass through a filtering layer:
- **Algorithm**: Exponential Moving Average (EMA).
- **Logic**: `Smoothed = α * New + (1 - α) * Previous`.
- **Default α**: `0.3` (Configurable via CLI).

### 4. Stress Model (`StressModel`)
The decision-making engine of the system:
- **Weighted Fusion**: Features are combined using a weighted sum.
  - *Lip Tension (0.30)*: Highest weight, as it is a primary indicator of suppressed stress.
  - *Eyebrow Raise (0.20)* & *Blink Rate (0.20)*: Secondary physiological indicators.
  - *Head Nod (0.15)* & *Symmetry (0.15)*: Subtle behavioral markers.
- **Categorization**:
  - `0.00 - 0.34`: **Calm**
  - `0.35 - 0.64`: **Slight Stress**
  - `0.65 - 1.00`: **High Stress**

---

## 💾 Data & Logging Systems

### Data Logger (`DataLogger`)
- **Format**: CSV.
- **Content**: Timestamped per-frame records of every facial feature and the resulting stress score.
- **Purpose**: Post-session review and behavioral audit.

### Landmark Recorder (`LandmarkRecorder`)
- **Format**: NumPy Binary (`.npy`).
- **Data**: Flattened vectors of 1,434 values per frame.
- **Purpose**: Generating high-quality datasets for training custom deep learning models (RNNs/LSTMs).

---

## 📊 Offline Analysis (`SessionAnalyzer`)
The backend includes a dedicated analysis suite to process recorded sessions:
- Reconstructs landmarks from `.npy` files.
- Re-runs the inference pipeline with full visualization.
- Generates temporal trend graphs using Matplotlib to identify specific moments of high psychological load.

---

## 🛠️ Technology Stack
- **OpenCV**: Video I/O and GUI rendering.
- **MediaPipe**: Computer Vision and Landmark extraction.
- **NumPy**: Matrix operations and high-speed feature math.
- **Matplotlib**: Statistical visualization.
- **Argparse**: Configurable CLI for different runtime environments (Headless vs. GUI).
