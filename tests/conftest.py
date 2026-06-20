import pytest
import numpy as np
import cv2
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.utils.signal_processing import SignalSmoother

@pytest.fixture
def mock_landmarks():
    """Returns a dummy landmark array of shape (478, 3)."""
    # Create simple structure: center nose tip at (0.5, 0.5, 0)
    landmarks = np.zeros((478, 3))
    
    # 1. NOSE_TIP
    landmarks[1] = [0.5, 0.5, 0.0]
    # 10. FACE_TOP
    landmarks[10] = [0.5, 0.2, 0.0]
    # 152. FACE_BOTTOM
    landmarks[152] = [0.5, 0.8, 0.0]
    
    # 105. L_EYEBROW
    landmarks[105] = [0.4, 0.35, 0.0]
    # 334. R_EYEBROW
    landmarks[334] = [0.6, 0.35, 0.0]
    # 159. L_EYE_UPPER
    landmarks[159] = [0.4, 0.4, 0.0]
    # 386. R_EYE_UPPER
    landmarks[386] = [0.6, 0.4, 0.0]
    
    # 61. MOUTH_L
    landmarks[61] = [0.4, 0.65, 0.0]
    # 291. MOUTH_R
    landmarks[291] = [0.6, 0.65, 0.0]
    # 13. MOUTH_T
    landmarks[13] = [0.5, 0.62, 0.0]
    # 14. MOUTH_B
    landmarks[14] = [0.5, 0.64, 0.0]
    
    # EAR Vertical/Horizontal Indices
    # L_EYE_V = [160, 144, 158, 153]
    landmarks[160] = [0.4, 0.39, 0.0]
    landmarks[144] = [0.4, 0.41, 0.0]
    landmarks[158] = [0.4, 0.39, 0.0]
    landmarks[153] = [0.4, 0.41, 0.0]
    # L_EYE_H = [33, 133]
    landmarks[33] = [0.38, 0.4, 0.0]
    landmarks[133] = [0.42, 0.4, 0.0]
    
    # R_EYE_V = [385, 380, 387, 373]
    landmarks[385] = [0.6, 0.39, 0.0]
    landmarks[380] = [0.6, 0.41, 0.0]
    landmarks[387] = [0.6, 0.39, 0.0]
    landmarks[373] = [0.6, 0.41, 0.0]
    # R_EYE_H = [362, 263]
    landmarks[362] = [0.58, 0.4, 0.0]
    landmarks[263] = [0.62, 0.4, 0.0]
    
    # Cheeks
    landmarks[234] = [0.3, 0.5, 0.0]
    landmarks[454] = [0.7, 0.5, 0.0]
    
    return landmarks

@pytest.fixture
def mock_frame():
    """Returns a dummy black image of size 640x480."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def feature_engineer():
    return FeatureEngineer()

@pytest.fixture
def stress_model():
    return StressModel(window_size=5)

@pytest.fixture
def signal_smoother():
    return SignalSmoother(alpha=0.3)
