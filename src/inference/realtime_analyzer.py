import cv2
import numpy as np
import time
import os
import argparse

from src.inference.face_landmark_detector import FaceLandmarkDetector
from src.inference.feature_engineering import FeatureEngineer
from src.inference.stress_model import StressModel
from src.inference.dashboard import Dashboard
from src.utils.data_logger import DataLogger
from src.utils.signal_processing import SignalSmoother
from src.utils.landmark_recorder import LandmarkRecorder
from src.utils.logger import setup_logging, get_logger
from src.utils.exceptions import CameraNotFoundError

setup_logging()
log = get_logger("realtime_analyzer")

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
    log.info("Initializing components...")
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
        raise CameraNotFoundError(f"Could not open webcam at index {args.camera_index}.")

    log.info(f"Starting Analyzer (Display: {not args.no_display}, Verbose: {args.verbose}, Record: {args.record_landmarks})...")
    log.info("Press 'q' in the window (if active) or Ctrl+C in terminal to quit.")
    
    start_time = time.time()
    prev_landmarks = None
    
    # FPS Tracking
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
                
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
                    log.info("-" * 30)
                    for k in features.keys():
                        raw_val = raw_features.get(k, 0.0)
                        smooth_val = features.get(k, 0.0)
                        log.info(f"{k.replace('_', ' ').title()}: Raw={raw_val:.3f}, Smooth={smooth_val:.3f}")
                    log.info(f"Stress Score: {stress_info['stress_score']:.2f}")
                    log.info(f"Level: {stress_info['stress_level']}")
                
                # Optional: Draw landmarks on the camera frame before dashboarding
                if not args.no_display:
                    frame = detector.draw_landmarks(frame, result)
                
                # Update state for next frame
                prev_landmarks = landmarks
                
            # 4. Render and Display
            if not args.no_display:
                # Put FPS text on the frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                display_frame = dashboard.render(frame, features, stress_info)
                cv2.imshow("AI Micro Expression Analyzer - Stress Detector", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, we can add a small sleep to save CPU if needed
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
            
    # 5. Cleanup
    cap.release()
    detector.close()
    if not args.no_display:
        cv2.destroyAllWindows()
        
    # Explicitly close logging session
    logger.close_session()
        
    # Save landmark recording session if enabled
    if recorder:
        recorder.save_session()
        
    log.info("Analyzer closed.")

if __name__ == "__main__":
    main()
