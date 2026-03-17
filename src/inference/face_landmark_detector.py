import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

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
        
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def _ensure_model_exists(self):
        """Downloads the model if it's missing."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Model not found at {self.model_path}. Downloading...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def extract_landmarks(self, frame, timestamp_ms):
        """
        Detects facial landmarks and returns them as a NumPy array.
        Args:
            frame: OpenCV BGR image.
            timestamp_ms: Current timestamp in milliseconds.
        Returns:
            np.ndarray: Array of shape (478, 3) or None if no face detected.
            result: MediaPipe FaceLandmarkerResult object.
        """
        # Convert BGR to RGB and then to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_landmarks:
            return None, result
        
        # Extract the first face's landmarks
        face_landmarks = result.face_landmarks[0]
        
        # Convert to NumPy array (478 landmarks)
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
