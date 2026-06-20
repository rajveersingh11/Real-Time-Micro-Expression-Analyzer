import csv
import os
from datetime import datetime
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("data_logger")

class DataLogger:
    """
    Handles logging of session data including facial metrics and stress scores.
    Saves data to CSV files in the logs/ directory.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.file_path = None
        self.file_handle = None
        self.writer = None
        self.fieldnames = [
            "timestamp",
            "eyebrow_raise",
            "lip_tension",
            "blink_rate",
            "head_nod",
            "symmetry",
            "stress_score",
            "stress_level"
        ]
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def start_session(self):
        """Creates a new CSV file with a timestamped name, writes the header, and keeps it open."""
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"session_{timestamp_str}.csv"
        self.file_path = os.path.join(self.log_dir, filename)
        
        try:
            self.file_handle = open(self.file_path, mode='w', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.file_handle.flush()
            logger.info(f"Session logging started: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to open session log file {self.file_path}: {e}")
            self.file_handle = None
            self.writer = None
            
        return self.file_path

    def log_frame(self, features, stress_result):
        """
        Logs a single frame's data to the CSV file.
        Args:
            features (dict): Facial features from FeatureEngineer.
            stress_result (dict): Stress scores from StressModel.
        """
        if self.file_path is None or self.file_handle is None or self.writer is None or not features or not stress_result:
            return

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "eyebrow_raise": f"{features.get('eyebrow_raise', 0.0):.4f}",
            "lip_tension": f"{features.get('lip_tension', 0.0):.4f}",
            "blink_rate": f"{features.get('blink_intensity', 0.0):.4f}",
            "head_nod": f"{features.get('head_nod', 0.0):.4f}",
            "symmetry": f"{features.get('symmetry_delta', 0.0):.4f}",
            "stress_score": f"{stress_result.get('stress_score', 0.0):.4f}",
            "stress_level": stress_result.get('stress_level', 'Unknown')
        }

        try:
            self.writer.writerow(row)
            self.file_handle.flush()
        except Exception as e:
            logger.error(f"Failed to write log row: {e}")

    def close_session(self):
        """Closes the CSV log file handle."""
        self._close_session_internal(log=True)

    def _close_session_internal(self, log=True):
        if hasattr(self, 'file_handle') and self.file_handle is not None:
            try:
                self.file_handle.close()
                if log:
                    logger.info("Session log file closed successfully.")
            except Exception as e:
                if log:
                    logger.error(f"Error closing session log file: {e}")
            finally:
                self.file_handle = None
                self.writer = None

    def __del__(self):
        self._close_session_internal(log=False)
