import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from src.utils.logger import setup_logging, get_logger
from src.utils.exceptions import ModelLoadError

setup_logging()
logger = get_logger("face_landmark_detector")

class FaceLandmarkDetector:
    """
    Facial landmark detection using MediaPipe Face Landmarker (Tasks API).
    Extracts 478 landmarks (including iris) and converts them to feature vectors.
    """
    def __init__(self, 
                 model_path="models/face_landmarker.task",
                 num_faces=1,
                 min_face_detection_confidence=0.5,
                 min_face_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.model_path = model_path
        self._ensure_model_exists()

        # Initialize MediaPipe Task
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        try:
            self.landmarker = FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize MediaPipe FaceLandmarker: {e}") from e

    def _ensure_model_exists(self):
        """Downloads the model with timeout and retry logic if missing."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            return
            
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        logger.info(f"Model not found at {self.model_path}. Downloading from {url}...")
        
        max_retries = 3
        timeout = 15
        
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
                logger.info("Download complete.")
                return
            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    raise ModelLoadError(f"Failed to download MediaPipe model after {max_retries} attempts: {e}") from e

    def close(self):
        """Clean up the landmarker resources."""
        self._close_internal(log=True)

    def _close_internal(self, log=True):
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            try:
                self.landmarker.close()
                if log:
                    logger.info("FaceLandmarker closed successfully.")
            except Exception as e:
                if log:
                    logger.error(f"Error closing FaceLandmarker: {e}")
            finally:
                self.landmarker = None

    def __del__(self):
        self._close_internal(log=False)

    def extract_landmarks(self, frame, timestamp_ms, return_multi=False):
        """
        Detects facial landmarks and returns them as a NumPy array or list of arrays.
        Args:
            frame: OpenCV BGR image.
            timestamp_ms: Current timestamp in milliseconds.
            return_multi: If True, returns a list of NumPy arrays (one per detected face).
                          If False, returns a single NumPy array for the first face.
        Returns:
            np.ndarray or list: Array of shape (478, 3) or list of such arrays (or None).
            result: MediaPipe FaceLandmarkerResult object.
        """
        # Convert BGR to RGB and then to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_landmarks:
            return None, result
        
        if return_multi:
            all_faces = []
            for face_landmarks in result.face_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks
                ])
                all_faces.append(landmarks)
            return all_faces, result
        else:
            # Extract the first face's landmarks
            face_landmarks = result.face_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks
            ])
            return landmarks, result

    def landmarks_to_vector(self, landmarks):
        """Flattens landmarks into a feature vector."""
        if landmarks is None:
            return None
        return landmarks.flatten()

    def draw_landmarks(self, frame, result):
        """Draws face landmarks on the frame using a manual drawing loop."""
        if not result.face_landmarks:
            return frame
            
        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
        return frame

if __name__ == "__main__":
    detector = FaceLandmarkDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam... Press 'q' to exit.")
    
    import time
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Get timestamp in ms
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        # Extract landmarks
        landmarks, result = detector.extract_landmarks(frame, timestamp_ms)
        
        if landmarks is not None:
            # Draw landmarks
            frame = detector.draw_landmarks(frame, result)

        cv2.imshow('MediaPipe Face Landmarker', frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
