import cv2
import numpy as np
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.inference.face_landmark_detector import FaceLandmarkDetector
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.utils.signal_processing import SignalSmoother
from src.utils.logger import get_logger

logger = get_logger("api_stream")
router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    # Initialize components per connection to maintain temporal history
    try:
        detector = FaceLandmarkDetector()
        feature_engineer = FeatureEngineer()
        smoother = SignalSmoother(alpha=0.3)
        stress_model = StressModel(window_size=15)
    except Exception as e:
        logger.error(f"Failed to initialize analysis components for WebSocket connection: {e}")
        await websocket.close(code=1011, reason="Failed to initialize models.")
        return
        
    prev_landmarks = None
    frame_count = 0
    
    try:
        while True:
            message = await websocket.receive()
            
            frame_bytes = None
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    image_data = data.get("image", "")
                    if "," in image_data:
                        image_data = image_data.split(",")[1]
                    frame_bytes = base64.b64decode(image_data)
                except Exception as e:
                    try:
                        frame_bytes = base64.b64decode(message["text"])
                    except Exception:
                        await websocket.send_json({"error": f"Invalid base64 payload: {str(e)}"})
                        continue
                        
            if frame_bytes is None:
                continue
                
            # Decode frame
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Failed to decode frame bytes."})
                continue
                
            frame_count += 1
            timestamp_ms = int(frame_count * (1000 / 30))
            
            # Process Frame
            landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
            
            if landmarks is None:
                await websocket.send_json({
                    "face_detected": False,
                    "features": None,
                    "stress": {
                        "stress_score": 0.0,
                        "stress_level": "No Face Detected"
                    }
                })
                continue
                
            # Raw features
            raw_features = feature_engineer.get_features(landmarks, prev_landmarks)
            # Smooth features
            features = smoother.smooth_features(raw_features)
            # Predict stress
            stress_info = stress_model.predict(features)
            
            # Update state
            prev_landmarks = landmarks
            
            # Send analysis back to client
            await websocket.send_json({
                "face_detected": True,
                "features": {
                    "eyebrow_raise": round(features.get("eyebrow_raise", 0.0), 4),
                    "lip_tension": round(features.get("lip_tension", 0.0), 4),
                    "blink_intensity": round(features.get("blink_intensity", 0.0), 4),
                    "head_nod": round(features.get("head_nod", 0.0), 4),
                    "symmetry_delta": round(features.get("symmetry_delta", 0.0), 4)
                },
                "stress": {
                    "stress_score": stress_info.get("stress_score", 0.0),
                    "stress_level": stress_info.get("stress_level", "Unknown")
                }
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection disconnected by client.")
    except Exception as e:
        logger.error(f"Error handling WebSocket stream: {e}")
    finally:
        # Resource cleanup
        if detector is not None:
            detector.close()
