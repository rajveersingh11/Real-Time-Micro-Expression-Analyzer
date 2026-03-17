import cv2
import numpy as np
import time
import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference.face_landmark_detector import FaceLandmarkDetector
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.inference.dashboard import Dashboard
from src.utils.data_logger import DataLogger
from src.utils.signal_processing import SignalSmoother
from src.utils.landmark_recorder import LandmarkRecorder

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Micro-Expression Analyzer - Real-Time Stress Detection")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without showing the UI")
    parser.add_argument("--verbose", action="store_true", help="Print detailed metrics to the terminal")
    parser.add_argument("--log-path", type=str, default=None, help="Custom path for the session log CSV")
    parser.add_argument("--alpha", type=float, default=0.3, help="Smoothing factor (0.0 to 1.0, lower is smoother)")
    parser.add_argument("--record-landmarks", action="store_true", help="Record landmark sequences as .npy files for training")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Initialize System Components
    print("Initializing components...")
    detector = FaceLandmarkDetector()
    feature_engineer = FeatureEngineer()
    stress_model = StressModel(window_size=15)
    dashboard = Dashboard()
    smoother = SignalSmoother(alpha=args.alpha)
    recorder = LandmarkRecorder() if args.record_landmarks else None
    
    # Initialize logger with optional custom path
    log_dir = os.path.dirname(args.log_path) if args.log_path else "logs"
    logger = DataLogger(log_dir=log_dir)
    session_file = logger.start_session()
    
    # 2. Setup Webcam Capture
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {args.camera_index}.")
        return

    print(f"Starting Analyzer (Display: {not args.no_display}, Verbose: {args.verbose}, Record: {args.record_landmarks})...")
    print("Press 'q' in the window (if active) or Ctrl+C in terminal to quit.")
    
    start_time = time.time()
    prev_landmarks = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get timestamp in ms for MediaPipe Tasks API
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # 3. Process Frame
            landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
            
            stress_info = {"stress_score": 0.0, "stress_level": "No Face Detected"}
            features = {}
            
            if landmarks is not None:
                # Extract Landmarks Vector (Flattened)
                landmark_vector = detector.landmarks_to_vector(landmarks)
                
                # Record landmarks if enabled
                if recorder:
                    recorder.record(landmark_vector)
                
                # Extract Features (Raw)
                raw_features = feature_engineer.get_features(landmarks, prev_landmarks)
                
                # Apply Temporal Smoothing (EMA)
                features = smoother.smooth_features(raw_features)
                
                # Predict Stress
                stress_info = stress_model.predict(features)
                
                # Log current frame data
                logger.log_frame(features, stress_info)
                
                # Verbose output
                if args.verbose:
                    print("-" * 30)
                    for k in features.keys():
                        raw_val = raw_features.get(k, 0.0)
                        smooth_val = features.get(k, 0.0)
                        print(f"{k.replace('_', ' ').title()}: Raw={raw_val:.3f}, Smooth={smooth_val:.3f}")
                    print(f"Stress Score: {stress_info['stress_score']:.2f}")
                    print(f"Level: {stress_info['stress_level']}")
                
                # Optional: Draw landmarks on the camera frame before dashboarding
                if not args.no_display:
                    frame = detector.draw_landmarks(frame, result)
                
                # Update state for next frame
                prev_landmarks = landmarks
                
            # 4. Render and Display
            if not args.no_display:
                display_frame = dashboard.render(frame, features, stress_info)
                cv2.imshow("AI Micro Expression Analyzer - Stress Detector", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, we can add a small sleep to save CPU if needed
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
            
    # 5. Cleanup
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
        
    # Save landmark recording session if enabled
    if recorder:
        recorder.save_session()
        
    print("Analyzer closed.")

if __name__ == "__main__":
    main()
