import numpy as np
import os
from datetime import datetime

class LandmarkRecorder:
    """
    Records sequences of facial landmark vectors for dataset generation and model training.
    Each frame is stored as a flattened vector (478 * 3 = 1434 values).
    """
    def __init__(self, output_dir="data/processed/landmark_sessions"):
        self.output_dir = output_dir
        self.frames = []
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def record(self, landmark_vector):
        """
        Appends a flattened landmark vector to the session buffer.
        Args:
            landmark_vector (np.ndarray): Flattened vector of shape (1434,).
        """
        if landmark_vector is not None:
            # Ensure it's a numpy array and copy it
            self.frames.append(np.array(landmark_vector, copy=True))

    def save_session(self):
        """
        Converts the frame buffer to a NumPy array and saves it as a .npy file.
        Returns:
            str: Path to the saved file or None if no frames were recorded.
        """
        if not self.frames:
            print("No frames recorded. Skipping save.")
            return None
            
        # Convert list of vectors to (N, 1434) array
        session_data = np.stack(self.frames)
        
        # Generate timestamped filename
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"session_{timestamp_str}.npy"
        file_path = os.path.join(self.output_dir, filename)
        
        # Save as .npy
        np.save(file_path, session_data)
        
        print(f"Landmark session saved: {file_path} (Frames: {len(self.frames)}, Shape: {session_data.shape})")
        
        # Clear buffer after saving
        self.frames = []
        
        return file_path

if __name__ == "__main__":
    # Quick test
    recorder = LandmarkRecorder(output_dir="logs/test_landmarks")
    for _ in range(10):
        dummy_vector = np.random.rand(1434)
        recorder.record(dummy_vector)
    recorder.save_session()
