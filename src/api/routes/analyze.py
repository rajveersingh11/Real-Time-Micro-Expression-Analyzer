import cv2
import numpy as np
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from src.api.schemas import AnalysisResponse, FeatureResponse, StressResponse
from src.inference.face_landmark_detector import FaceLandmarkDetector
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.utils.signal_processing import SignalSmoother
from src.utils.logger import get_logger

logger = get_logger("api_analyze")
router = APIRouter()

# Initialize core model components once at start
try:
    detector = FaceLandmarkDetector()
    feature_engineer = FeatureEngineer()
    stress_model = StressModel(window_size=1)  # No temporal window smoothing needed for static frames
    smoother = SignalSmoother()
except Exception as e:
    logger.error(f"Failed to initialize analysis components for API: {e}")
    detector = None
    feature_engineer = None
    stress_model = None
    smoother = None

@router.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_frame(file: UploadFile = File(...)):
    """
    Upload an image frame and extract facial landmarks, metrics, and stress predictions.
    """
    if detector is None or feature_engineer is None or stress_model is None or smoother is None:
        raise HTTPException(status_code=503, detail="Model components are not available.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read file bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        timestamp_ms = int(time.time() * 1000)
        landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
        
        if landmarks is None:
            return AnalysisResponse(
                face_detected=False,
                features=None,
                stress=StressResponse(
                    stress_score=0.0,
                    stress_level="No Face Detected"
                )
            )
            
        # Get raw features
        raw_features = feature_engineer.get_features(landmarks)
        # Smooth features
        features = smoother.smooth_features(raw_features)
        # Predict stress
        stress_info = stress_model.predict(features)
        
        return AnalysisResponse(
            face_detected=True,
            features=FeatureResponse(
                eyebrow_raise=features.get("eyebrow_raise", 0.0),
                lip_tension=features.get("lip_tension", 0.0),
                blink_intensity=features.get("blink_intensity", 0.0),
                head_nod=features.get("head_nod", 0.0),
                symmetry_delta=features.get("symmetry_delta", 0.0)
            ),
            stress=StressResponse(
                stress_score=stress_info.get("stress_score", 0.0),
                stress_level=stress_info.get("stress_level", "Unknown")
            )
        )
    except Exception as e:
        logger.error(f"Error during API frame analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
