import pytest
from src.utils.signal_processing import SignalSmoother
from src.utils.exceptions import ConfigurationError

def test_alpha_validation():
    # Valid alpha
    smoother = SignalSmoother(alpha=0.5)
    assert smoother.alpha == 0.5
    
    # Invalid alphas should raise ConfigurationError
    with pytest.raises(ConfigurationError):
        SignalSmoother(alpha=1.5)
        
    with pytest.raises(ConfigurationError):
        SignalSmoother(alpha=-0.1)

def test_ema_smooth(signal_smoother):
    v1 = signal_smoother.ema_smooth("test_feat", 0.5)
    assert v1 == 0.5
    
    v2 = signal_smoother.ema_smooth("test_feat", 1.0)
    # EMA formula with alpha=0.3: 0.3 * 1.0 + 0.7 * 0.5 = 0.65
    assert abs(v2 - 0.65) < 1e-6

def test_smooth_features(signal_smoother):
    features = {
        "eyebrow_raise": 0.5,
        "lip_tension": 0.5
    }
    smoothed = signal_smoother.smooth_features(features)
    assert set(smoothed.keys()) == set(features.keys())
    assert smoothed["eyebrow_raise"] == 0.5
