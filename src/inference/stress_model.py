import numpy as np
from collections import deque

class StressModel:
    """
    Estimates stress levels by fusing facial behavior features.
    Provides temporal smoothing and categorical classification.
    """
    def __init__(self, window_size=10):
        # Weights for feature fusion
        self.weights = {
            "eyebrow_raise": 0.20,
            "lip_tension": 0.30,
            "blink_intensity": 0.20, # Mapping blink_rate to the key used in feature_engineering.py
            "head_nod": 0.15,
            "symmetry_delta": 0.15   # Mapping symmetry to the key used in feature_engineering.py
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
            score (float): Smoothed stress score.
        Returns:
            str: Stress level classification ('Calm', 'Slight Stress', 'High Stress').
        """
        if score < 0.35:
            return "Calm"
        elif score < 0.65:
            return "Slight Stress"
        else:
            return "High Stress"

    def predict(self, features):
        """
        Full prediction pipeline: compute -> smooth -> classify.
        Args:
            features (dict): Dictionary of facial features.
        Returns:
            dict: Dictionary containing score and level.
        """
        raw_score = self.compute_score(features)
        smoothed_score = self.smooth_score(raw_score)
        level = self.classify_stress(smoothed_score)
        
        return {
            "stress_score": round(smoothed_score, 4),
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
