# AI-MicroExpression-Analyzer Architecture

This document describes the high-level architecture, module decomposition, and data flow of the Real-Time Micro-Expression and Stress Analyzer.

## System Architecture Flow

The system runs a linear, high-frequency, pipeline-based data transformation:

```
  +--------------+
  | Video/Camera | (Capture 30+ FPS BGR frame)
  +-------+------+
          |
          v
  +-------+------+
  | MediaPipe    | (Extract 478 3D landmarks in VIDEO mode)
  +-------+------+
          |
          v
  +-------+------+
  | Feature      | (Compute ratios: Eyebrow raise, Lip tension,
  | Engineering  |  EAR blink intensity, Head nod, Cheek symmetry)
  +-------+------+
          |
          v
  +-------+------+
  | Signal       | (Filter noise using Exponential Moving
  | Smoother     |  Average with configured alpha)
  +-------+------+
          |
          v
  +-------+------+
  | Stress       | (Fuse features using data-validated weights
  | Model        |  and classify Calm/Slight Stress/High Stress)
  +-------+------+
          |
          +-----------------------------+
          |                             |
          v                             v
  +-------+------+              +-------+------+
  | Data Logger  |              | Web/OpenCV   | (Visualize metrics
  | (Log to CSV) |              | Dashboard    |  and real-time plots)
  +--------------+              +--------------+
```

---

## Component Decomposition

### 1. Computer Vision Layer
- **FaceLandmarkDetector**: Interacts with the MediaPipe Tasks API. It manages local model downloading with retry/timeout safety, processes video frames, extracts 478 facial landmarks, and ensures proper cleanup of memory resources on deletion.

### 2. Feature Extraction Layer
- **FeatureEngineer**: Translates raw geometric points into human-interpretable physiological markers.
  - **Eyebrow Raise**: Distance between eyebrow and upper eyelid normalized by face height.
  - **Lip Tension**: Width-to-height ratio of mouth boundaries.
  - **Blink Intensity**: Eye Aspect Ratio (EAR) based on vertical and horizontal eyelid metrics.
  - **Head Nod**: Vertical delta shifts of the nose tip relative to the previous frame.
  - **Facial Symmetry**: Bilateral asymmetry index computed by comparing distances from cheeks to nose tip.

### 3. Smoothing and Filtering Layer
- **SignalSmoother**: Noise and micro-tremor reduction using Exponential Moving Average (EMA). It maintains a rolling buffer history for each signal.

### 4. Stress Estimation Layer
- **StressModel**: Performs weighted feature fusion on the smoothed signal inputs to produce a continuous stress score (0.0 to 1.0) and translates it into discrete levels (`Calm`, `Slight Stress`, `High Stress`).

### 5. API Layer
- **FastAPI / Uvicorn Server**: Offers a scalable HTTP backend.
  - `GET /health` for diagnostics.
  - `POST /analyze` for image frame parsing.
  - `WS /ws/stream` for continuous real-time video stream processing via binary or base64 frame encoding.

---

## Configuration Management

Settings are managed via a centralized YAML configuration system:
- **default.yaml**: Contains parameters for inference thresholds, stress fusion weights, training epochs, optimizer types, learning rate schedules, and logging verbosity.
- **config.py**: Handles safe parsing, fallback resolutions, and path checking.
