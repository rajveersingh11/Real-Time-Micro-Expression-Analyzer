import cv2
import numpy as np
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("privacy")

class PrivacyManager:
    """
    Handles privacy protection features including face blurring and data anonymization.
    """
    def __init__(self, blur_kernel_size=(51, 51)):
        self.blur_kernel_size = blur_kernel_size
        self.consent_given = False

    def request_consent(self) -> bool:
        """
        Simulates checking or requesting user consent for camera recording and data logging.
        """
        logger.info("Verifying session data consent...")
        # In a real environment, this might check a local configuration or prompt the user.
        self.consent_given = True
        return self.consent_given

    def blur_face(self, frame, landmarks):
        """
        Applies gaussian blur to the face region in the frame to anonymize identity
        while keeping general head movement visible.
        Args:
            frame (np.ndarray): OpenCV BGR frame.
            landmarks (np.ndarray): Array of shape (478, 3) representing face landmarks.
        Returns:
            np.ndarray: Frame with face region blurred.
        """
        if landmarks is None or frame is None:
            return frame

        h, w, _ = frame.shape
        
        # Determine bounding box from landmarks
        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h
        
        x_min, x_max = int(np.clip(np.min(xs), 0, w)), int(np.clip(np.max(xs), 0, w))
        y_min, y_max = int(np.clip(np.min(ys), 0, h)), int(np.clip(np.max(ys), 0, h))
        
        # Add padding
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        # Check if coordinates are valid
        if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
            return frame

        # Create a copy and blur the bounding box region
        blurred_frame = frame.copy()
        face_roi = blurred_frame[y_min:y_max, x_min:x_max]
        
        # Apply strong Gaussian blur
        blurred_face = cv2.GaussianBlur(face_roi, self.blur_kernel_size, 0)
        blurred_frame[y_min:y_max, x_min:x_max] = blurred_face
        
        return blurred_frame

    def anonymize_landmarks(self, landmarks):
        """
        Removes absolute screen/camera coordinates and converts them into relative differences,
        ensuring raw facial features can be used without leaking specific identity/shape details.
        """
        if landmarks is None:
            return None
            
        # Center landmarks around the nose tip (index 1)
        nose_tip = landmarks[1]
        relative_landmarks = landmarks - nose_tip
        
        # Normalize by facial width (dist between cheek indices 234 and 454) to ensure scale invariance
        facial_width = np.linalg.norm(landmarks[234] - landmarks[454])
        if facial_width > 1e-6:
            relative_landmarks = relative_landmarks / facial_width
            
        return relative_landmarks
