import cv2
import numpy as np

class Dashboard:
    """
    Visual dashboard for AI-MicroExpression-Analyzer.
    Displays stress levels and facial feature metrics using progress bars.
    """
    def __init__(self, sidebar_width=250):
        self.sidebar_width = sidebar_width
        
        # Colors (BGR)
        self.COLOR_CALM = (0, 255, 0)      # Green
        self.COLOR_SLIGHT = (0, 255, 255)  # Yellow
        self.COLOR_HIGH = (0, 0, 255)      # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_BG = (30, 30, 30)       # Dark Gray

    def draw_progress_bar(self, frame, x, y, value, label, color):
        """Draws a horizontal progress bar with a label."""
        bar_width = 200
        bar_height = 15
        
        # Draw label
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        
        # Draw background bar
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Draw filled bar
        fill_w = int(value * bar_width)
        cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_height), color, -1)
        
        # Draw percentage/value
        cv2.putText(frame, f"{int(value * 100)}%", (x + bar_width + 5, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)

    def render(self, frame, features, stress_info):
        """
        Renders the dashboard on the frame.
        Expands the frame to include a sidebar.
        """
        h, w, _ = frame.shape
        # Create a new canvas with sidebar
        canvas = np.zeros((h, w + self.sidebar_width, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = self.COLOR_BG
        
        # Determine Stress Color
        level = stress_info.get("stress_level", "Unknown")
        score = stress_info.get("stress_score", 0.0)
        
        if level == "Calm":
            current_color = self.COLOR_CALM
        elif level == "Slight Stress":
            current_color = self.COLOR_SLIGHT
        elif level == "High Stress":
            current_color = self.COLOR_HIGH
        else:
            current_color = (128, 128, 128)

        # 1. Stress Indicator Banner (Top of Sidebar)
        cv2.putText(canvas, "STRESS ANALYSIS", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.line(canvas, (w + 10, 55), (w + self.sidebar_width - 10, 55), (100, 100, 100), 1)
        
        cv2.putText(canvas, f"Level: {level}", (w + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)
        cv2.putText(canvas, f"Score: {score:.2f}", (w + 20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
        
        # 2. Feature Metrics (Progress Bars)
        if features:
            cv2.putText(canvas, "FACIAL METRICS", (w + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Metric List
            metrics = [
                ("Eyebrow Raise", features.get("eyebrow_raise", 0.0)),
                ("Lip Tension", features.get("lip_tension", 0.0)),
                ("Blink Rate", features.get("blink_intensity", 0.0)),
                ("Head Nod", features.get("head_nod", 0.0)),
                ("Symmetry", features.get("symmetry_delta", 0.0))
            ]
            
            start_y = 220
            for label, val in metrics:
                self.draw_progress_bar(canvas, w + 20, start_y, val, label, current_color)
                start_y += 50

        # 3. Color-Coded Border on Camera Feed
        cv2.rectangle(canvas, (0, 0), (w, h), current_color, 8)
        
        return canvas
