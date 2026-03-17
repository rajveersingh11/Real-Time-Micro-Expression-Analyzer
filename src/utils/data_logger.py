import csv
import os
from datetime import datetime

class DataLogger:
    """
    Handles logging of session data including facial metrics and stress scores.
    Saves data to CSV files in the logs/ directory.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.file_path = None
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
        """Creates a new CSV file with a timestamped name and writes the header."""
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"session_{timestamp_str}.csv"
        self.file_path = os.path.join(self.log_dir, filename)
        
        with open(self.file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            
        print(f"Session logging started: {self.file_path}")
        return self.file_path

    def log_frame(self, features, stress_result):
        """
        Logs a single frame's data to the CSV file.
        Args:
            features (dict): Facial features from FeatureEngineer.
            stress_result (dict): Stress scores from StressModel.
        """
        if self.file_path is None or not features or not stress_result:
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

        # Append to CSV
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
