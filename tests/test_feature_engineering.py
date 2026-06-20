import numpy as np
from src.inference.feature_engineering import FeatureEngineer

def test_eyebrow_raise(feature_engineer, mock_landmarks):
    val = feature_engineer.eyebrow_raise(mock_landmarks)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

def test_lip_tension(feature_engineer, mock_landmarks):
    val = feature_engineer.lip_tension(mock_landmarks)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

def test_blink_rate(feature_engineer, mock_landmarks):
    val = feature_engineer.blink_rate(mock_landmarks)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

def test_head_nod_intensity(feature_engineer, mock_landmarks):
    prev_landmarks = mock_landmarks.copy()
    prev_landmarks[1] = [0.5, 0.49, 0.0]  # shift nose tip vertically
    val = feature_engineer.head_nod_intensity(mock_landmarks, prev_landmarks)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

def test_symmetry_delta(feature_engineer, mock_landmarks):
    val = feature_engineer.symmetry_delta(mock_landmarks)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

def test_get_features(feature_engineer, mock_landmarks):
    features = feature_engineer.get_features(mock_landmarks)
    assert isinstance(features, dict)
    expected_keys = {"eyebrow_raise", "lip_tension", "blink_intensity", "head_nod", "symmetry_delta"}
    assert set(features.keys()) == expected_keys
    for k, v in features.items():
        assert 0.0 <= v <= 1.0
