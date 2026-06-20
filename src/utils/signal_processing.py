import numpy as np
from collections import deque
from src.utils.exceptions import ConfigurationError
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("signal_processing")

class SignalSmoother:
    """
    Provides temporal smoothing and noise filtering for facial behavior signals.
    Uses Exponential Moving Average (EMA) and rolling window buffers.
    """
    def __init__(self, alpha=0.3, window_size=10):
        if not (0.0 <= alpha <= 1.0):
            raise ConfigurationError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self.alpha = alpha
        self.window_size = window_size
        
        # Dictionary to store the previous smoothed value for each feature (for EMA)
        self.prev_smoothed = {}
        
        # Optional: Rolling window buffers for each feature (for future statistical analysis)
        self.buffers = {
            "eyebrow_raise": deque(maxlen=window_size),
            "lip_tension": deque(maxlen=window_size),
            "blink_intensity": deque(maxlen=window_size),
            "head_nod": deque(maxlen=window_size),
            "symmetry_delta": deque(maxlen=window_size)
        }

    def ema_smooth(self, feature_name, value):
        """
        Applies Exponential Moving Average (EMA) to a single feature.
        Formula: smoothed_value = alpha * new_value + (1 - alpha) * previous_value
        """
        if feature_name not in self.prev_smoothed:
            # First value: no previous smoothed value exists
            smoothed_value = value
        else:
            previous_value = self.prev_smoothed[feature_name]
            smoothed_value = self.alpha * value + (1.0 - self.alpha) * previous_value
            
        # Update state
        self.prev_smoothed[feature_name] = smoothed_value
        
        # Also update buffer
        if feature_name in self.buffers:
            self.buffers[feature_name].append(smoothed_value)
            
        return float(smoothed_value)

    def get_history(self, feature_name):
        """
        Returns the rolling history buffer for a specific feature.
        """
        if feature_name in self.buffers:
            return list(self.buffers[feature_name])
        return []

    def smooth_features(self, features):
        """
        Applies smoothing to all features in the provided dictionary.
        Returns a new dictionary with smoothed values.
        """
        if not features:
            return {}
            
        smoothed_features = {}
        for key, value in features.items():
            smoothed_features[key] = self.ema_smooth(key, value)
            
        return smoothed_features

if __name__ == "__main__":
    # Test with dummy data
    smoother = SignalSmoother(alpha=0.3)
    
    raw_values = [0.1, 0.5, 0.2, 0.8, 0.4]
    print(f"Alpha: {smoother.alpha}")
    print("Testing smoothing on 'eyebrow_raise':")
    for val in raw_values:
        smoothed = smoother.ema_smooth("eyebrow_raise", val)
        print(f"Raw: {val:.2f} -> Smoothed: {smoothed:.4f}")
