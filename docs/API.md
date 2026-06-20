# AI-MicroExpression-Analyzer API documentation

This document provides details for interacting with the HTTP and WebSocket endpoints of the analyzer API.

---

## 🚀 Running the API Server

The API server can be launched directly using the registered script:
```bash
mea-server
```
By default, the server listens at `http://localhost:8000`. You can access the interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

## 📌 Endpoint Summary

### 1. Health Check
* **Path**: `GET /health`
* **Description**: Verifies that the service is running and ML models are active.
* **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### 2. Static Frame Analysis
* **Path**: `POST /analyze`
* **Content-Type**: `multipart/form-data`
* **Description**: Upload a single image file to receive a static facial feature breakdown and stress calculation.
* **Request Params**:
  - `file`: Binary image file (JPEG, PNG).
* **Success Response (200 OK)**:
  ```json
  {
    "face_detected": true,
    "features": {
      "eyebrow_raise": 0.154,
      "lip_tension": 0.421,
      "blink_intensity": 0.082,
      "head_nod": 0.0,
      "symmetry_delta": 0.045
    },
    "stress": {
      "stress_score": 0.28,
      "stress_level": "Calm"
    }
  }
  ```
* **No Face Detected Response**:
  ```json
  {
    "face_detected": false,
    "features": null,
    "stress": {
      "stress_score": 0.0,
      "stress_level": "No Face Detected"
    }
  }
  ```

### 3. Real-Time WebSocket Streaming
* **Path**: `WS /ws/stream`
* **Description**: Establishes a persistent connection for sending sequential frames and receiving real-time predictions. The server maintains temporal smoothing history (EMA) for this specific connection.
* **Payload Types**:
  - **Binary**: raw frame bytes (JPEG or PNG).
  - **Text**: JSON object containing a base64 encoded frame:
    ```json
    {
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
    ```
* **Server Push Response (JSON)**:
  ```json
  {
    "face_detected": true,
    "features": {
      "eyebrow_raise": 0.1821,
      "lip_tension": 0.3842,
      "blink_intensity": 0.1023,
      "head_nod": 0.0234,
      "symmetry_delta": 0.0381
    },
    "stress": {
      "stress_score": 0.312,
      "stress_level": "Calm"
    }
  }
  ```
