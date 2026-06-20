import numpy as np
from collections import deque
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("stress_model")

class StressModel:
    """
    Estimates stress levels by fusing facial behavior features.
    Provides temporal smoothing and categorical classification.
    """
    def __init__(self, window_size=10, smooth_score_enabled=False):
        # Load weights and thresholds from config or use default fallback
        try:
            config = load_config()
            weights = config["stress"]["weights"]
            self.thresholds = config["stress"]["thresholds"]
            self.smooth_score_enabled = config["stress"].get("smooth_score", smooth_score_enabled)
        except Exception as e:
            logger.warning(f"Failed to load stress config, using defaults: {e}")
            weights = {
                "eyebrow_raise": 0.20,
                "lip_tension": 0.30,
                "blink_intensity": 0.20,
                "head_nod": 0.15,
                "facial_symmetry": 0.15
            }
            self.thresholds = [0.35, 0.65]
            self.smooth_score_enabled = smooth_score_enabled

        self.weights = {
            "eyebrow_raise": weights.get("eyebrow_raise", 0.20),
            "lip_tension": weights.get("lip_tension", 0.30),
            "blink_intensity": weights.get("blink_intensity", 0.20),
            "head_nod": weights.get("head_nod", 0.15),
            "symmetry_delta": weights.get("symmetry_delta", weights.get("facial_symmetry", 0.15))
        }
        
        # Temporal smoothing buffer
        self.score_history = deque(maxlen=window_size)

    def compute_score(self, features):
        """
        Calculates a raw stress score (0-1) using weighted fusion.
        Args:
            features (dict): Dictionary of normalized facial features.
        Returns:
            float: Raw stress score.
        """
        if not features:
            return 0.0
            
        score = 0.0
        for key, weight in self.weights.items():
            # Use .get() to handle potential missing keys gracefully
            score += features.get(key, 0.0) * weight
            
        return float(np.clip(score, 0, 1))

    def smooth_score(self, score):
        """
        Smooths the stress score using a rolling average.
        Args:
            score (float): Current raw stress score.
        Returns:
            float: Smoothed stress score.
        """
        self.score_history.append(score)
        return float(np.mean(self.score_history))

    def classify_stress(self, score):
        """
        Classifies the stress score into a categorical level.
        Args:
            score (float): Stress score.
        Returns:
            str: Stress level classification ('Calm', 'Slight Stress', 'High Stress').
        """
        if score < self.thresholds[0]:
            return "Calm"
        elif score < self.thresholds[1]:
            return "Slight Stress"
        else:
            return "High Stress"

    def predict(self, features):
        """
        Full prediction pipeline: compute -> optionally smooth -> classify.
        Args:
            features (dict): Dictionary of facial features.
        Returns:
            dict: Dictionary containing score and level.
        """
        raw_score = self.compute_score(features)
        
        if self.smooth_score_enabled:
            final_score = self.smooth_score(raw_score)
        else:
            final_score = raw_score
            
        level = self.classify_stress(final_score)
        
        return {
            "stress_score": round(final_score, 4),
            "stress_level": level
        }

if __name__ == "__main__":
    # Test with dummy features
    sm = StressModel()
    
    # Simulate a series of frames
    test_features = {
        "eyebrow_raise": 0.4,
        "lip_tension": 0.7,
        "blink_intensity": 0.2,
        "head_nod": 0.1,
        "symmetry_delta": 0.3
    }
    
    print("Simulating 5 frames of predictions:")
    for i in range(5):
        prediction = sm.predict(test_features)
        print(f"Frame {i+1}: {prediction}")
