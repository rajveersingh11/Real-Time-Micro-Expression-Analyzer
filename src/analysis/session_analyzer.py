import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.utils.signal_processing import SignalSmoother

def load_session(file_path):
    """Loads a .npy landmark session file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    data = np.load(file_path)
    print(f"Loaded session data from {file_path}")
    print(f"Session shape: {data.shape} (Frames, Landmarks*3)")
    return data

def vector_to_landmarks(vector):
    """Converts a flattened (1434,) vector back to (478, 3)."""
    return vector.reshape((478, 3))

def analyze_session(data):
    """
    Processes the session data frame-by-frame and computes stress metrics.
    """
    feature_engineer = FeatureEngineer()
    smoother = SignalSmoother(alpha=0.3)
    stress_model = StressModel(window_size=15)
    
    num_frames = data.shape[0]
    results = {
        "eyebrow_raise": [],
        "lip_tension": [],
        "blink_rate": [],
        "head_nod": [],
        "symmetry": [],
        "stress_score": []
    }
    
    prev_landmarks = None
    
    print(f"Analyzing {num_frames} frames...")
    for i in range(num_frames):
        vector = data[i]
        landmarks = vector_to_landmarks(vector)
        
        # 1. Extract Features
        raw_features = feature_engineer.get_features(landmarks, prev_landmarks)
        
        # 2. Smooth Features
        smoothed_features = smoother.smooth_features(raw_features)
        
        # 3. Compute Stress
        stress_info = stress_model.predict(smoothed_features)
        
        # Store results
        results["eyebrow_raise"].append(smoothed_features.get("eyebrow_raise", 0.0))
        results["lip_tension"].append(smoothed_features.get("lip_tension", 0.0))
        results["blink_rate"].append(smoothed_features.get("blink_intensity", 0.0))
        results["head_nod"].append(smoothed_features.get("head_nod", 0.0))
        results["symmetry"].append(smoothed_features.get("symmetry_delta", 0.0))
        results["stress_score"].append(stress_info.get("stress_score", 0.0))
        
        prev_landmarks = landmarks
        
    return results

def plot_results(results, file_name):
    """Generates and displays plots for stress and feature trends."""
    frames = range(len(results["stress_score"]))
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Stress Score Over Time
    plt.subplot(2, 1, 1)
    plt.plot(frames, results["stress_score"], color='red', linewidth=2, label='Stress Score')
    plt.fill_between(frames, results["stress_score"], color='red', alpha=0.1)
    plt.title(f"Offline Stress Analysis: {file_name}")
    plt.xlabel("Frame Number")
    plt.ylabel("Stress Score (0.0 - 1.0)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Plot 2: Facial Feature Trends
    plt.subplot(2, 1, 2)
    plt.plot(frames, results["eyebrow_raise"], label='Eyebrow Raise', alpha=0.8)
    plt.plot(frames, results["lip_tension"], label='Lip Tension', alpha=0.8)
    plt.plot(frames, results["blink_rate"], label='Blink Rate', alpha=0.8)
    plt.plot(frames, results["head_nod"], label='Head Nod', alpha=0.8)
    plt.plot(frames, results["symmetry"], label='Symmetry', alpha=0.8)
    
    plt.title("Facial Feature Trends Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Normalized Feature Intensity")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Offline Session Analyzer for AI-MicroExpression-Analyzer")
    parser.add_argument("file_path", type=str, help="Path to the .npy landmark session file")
    args = parser.parse_args()
    
    # 1. Load Data
    data = load_session(args.file_path)
    if data is None:
        return
    
    # 2. Analyze
    results = analyze_session(data)
    
    # 3. Visualize
    plot_results(results, os.path.basename(args.file_path))

if __name__ == "__main__":
    main()
