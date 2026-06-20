import numpy as np
import os
from datetime import datetime
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("landmark_recorder")

class LandmarkRecorder:
    """
    Records sequences of facial landmark vectors for dataset generation and model training.
    Each frame is stored as a flattened vector (478 * 3 = 1434 values).
    """
    def __init__(self, output_dir="data/processed/landmark_sessions", max_buffer_size=10000):
        self.output_dir = output_dir
        self.max_buffer_size = max_buffer_size
        self.frames = []
        self.chunk_paths = []
        self.session_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.part_num = 1
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def record(self, landmark_vector):
        """
        Appends a flattened landmark vector to the session buffer.
        Saves as a chunk to disk if the buffer exceeds max_buffer_size.
        Args:
            landmark_vector (np.ndarray): Flattened vector of shape (1434,).
        """
        if landmark_vector is not None:
            # Ensure it's a numpy array and copy it
            self.frames.append(np.array(landmark_vector, copy=True))
            if len(self.frames) >= self.max_buffer_size:
                self._save_chunk()

    def _save_chunk(self):
        """Saves the current frame buffer as a chunk file to disk and clears the buffer."""
        if not self.frames:
            return
        session_data = np.stack(self.frames)
        filename = f"session_{self.session_id}_part{self.part_num}.npy"
        file_path = os.path.join(self.output_dir, filename)
        np.save(file_path, session_data)
        logger.info(f"Landmark chunk saved: {file_path} (Frames: {len(self.frames)})")
        self.chunk_paths.append(file_path)
        self.frames = []
        self.part_num += 1

    def save_session(self):
        """
        Saves any remaining frames, merges all chunks if chunked, and returns final file path.
        """
        if not self.frames and not self.chunk_paths:
            logger.warning("No frames recorded. Skipping save.")
            return None
            
        # Save remaining frames as last chunk if there are any
        if self.frames:
            self._save_chunk()
            
        final_filename = f"session_{self.session_id}.npy"
        final_file_path = os.path.join(self.output_dir, final_filename)
        
        if len(self.chunk_paths) == 1:
            try:
                os.rename(self.chunk_paths[0], final_file_path)
                logger.info(f"Landmark session saved: {final_file_path}")
                self.chunk_paths = []
                return final_file_path
            except Exception as e:
                logger.error(f"Failed to rename chunk file to final name: {e}")
                return self.chunk_paths[0]
                
        # Merge chunk files
        try:
            logger.info(f"Merging {len(self.chunk_paths)} chunks into {final_file_path}...")
            merged_data = []
            for path in self.chunk_paths:
                merged_data.append(np.load(path))
            final_data = np.concatenate(merged_data, axis=0)
            np.save(final_file_path, final_data)
            logger.info(f"Merged landmark session saved: {final_file_path} (Total Frames: {final_data.shape[0]})")
            
            # Clean up temporary chunk files
            for path in self.chunk_paths:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Failed to delete temp chunk file {path}: {e}")
            self.chunk_paths = []
            return final_file_path
        except Exception as e:
            logger.error(f"Failed to merge chunks: {e}")
            return ", ".join(self.chunk_paths)

if __name__ == "__main__":
    # Quick test
    recorder = LandmarkRecorder(output_dir="logs/test_landmarks")
    for _ in range(10):
        dummy_vector = np.random.rand(1434)
        recorder.record(dummy_vector)
    recorder.save_session()
