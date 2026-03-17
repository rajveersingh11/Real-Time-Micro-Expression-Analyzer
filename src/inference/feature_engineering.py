import numpy as np

class FeatureEngineer:
    """
    Converts raw 478 facial landmarks into interpretable behavior features.
    Features are normalized between 0 and 1.
    """
    def __init__(self):
        # MediaPipe Landmark Indices
        self.NOSE_TIP = 1
        self.FACE_TOP = 10
        self.FACE_BOTTOM = 152
        
        # Eyebrows and Eyes
        self.L_EYEBROW = 105
        self.R_EYEBROW = 334
        self.L_EYE_UPPER = 159
        self.R_EYE_UPPER = 386
        
        # Mouth
        self.MOUTH_L = 61
        self.MOUTH_R = 291
        self.MOUTH_T = 13
        self.MOUTH_B = 14
        
        # EAR Indices (Vertical pairs and horizontal ends)
        self.L_EYE_V = [160, 144, 158, 153]
        self.L_EYE_H = [33, 133]
        self.R_EYE_V = [385, 380, 387, 373]
        self.R_EYE_H = [362, 263]
        
        # Cheeks for symmetry
        self.L_CHEEK = 234
        self.R_CHEEK = 454

    def _dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def eyebrow_raise(self, landmarks):
        """Detects vertical movement of eyebrows relative to eye region."""
        face_height = self._dist(landmarks[self.FACE_TOP], landmarks[self.FACE_BOTTOM])
        
        # Distance between eyebrow and upper eyelid
        l_dist = self._dist(landmarks[self.L_EYEBROW], landmarks[self.L_EYE_UPPER])
        r_dist = self._dist(landmarks[self.R_EYEBROW], landmarks[self.R_EYE_UPPER])
        
        avg_dist = (l_dist + r_dist) / 2.0
        # Normalization based on heuristic: 0.15 of face height is a significant raise
        score = avg_dist / (face_height * 0.15)
        return float(np.clip(score, 0, 1))

    def lip_tension(self, landmarks):
        """Detects compressed or tense lips using width/height ratio."""
        width = self._dist(landmarks[self.MOUTH_L], landmarks[self.MOUTH_R])
        height = self._dist(landmarks[self.MOUTH_T], landmarks[self.MOUTH_B])
        
        # High ratio indicates thin, tense lips or wide compression
        ratio = width / max(height, 1e-6)
        # Normalization: ratio of 10+ is typically very tense
        score = (ratio - 2.0) / 8.0 
        return float(np.clip(score, 0, 1))

    def blink_rate(self, landmarks):
        """Computes Eye Aspect Ratio (EAR) as a blink intensity indicator."""
        def get_ear(v_indices, h_indices):
            v1 = self._dist(landmarks[v_indices[0]], landmarks[v_indices[1]])
            v2 = self._dist(landmarks[v_indices[2]], landmarks[v_indices[3]])
            h = self._dist(landmarks[h_indices[0]], landmarks[h_indices[1]])
            return (v1 + v2) / (2.0 * max(h, 1e-6))

        l_ear = get_ear(self.L_EYE_V, self.L_EYE_H)
        r_ear = get_ear(self.R_EYE_V, self.R_EYE_H)
        avg_ear = (l_ear + r_ear) / 2.0
        
        # Blink intensity: 1.0 when eyes are closed (EAR ~0.0), 0.0 when open (EAR ~0.3)
        intensity = 1.0 - (avg_ear / 0.35)
        return float(np.clip(intensity, 0, 1))

    def head_nod_intensity(self, current_landmarks, previous_landmarks):
        """Detects vertical head movement intensity."""
        if previous_landmarks is None:
            return 0.0
            
        face_height = self._dist(current_landmarks[self.FACE_TOP], current_landmarks[self.FACE_BOTTOM])
        # Vertical movement of the nose tip
        movement = abs(current_landmarks[self.NOSE_TIP][1] - previous_landmarks[self.NOSE_TIP][1])
        
        # Normalization: movement of 5% face height is a clear nod
        score = movement / (face_height * 0.05)
        return float(np.clip(score, 0, 1))

    def symmetry_delta(self, landmarks):
        """Detects asymmetry between left and right face movement."""
        l_dist = self._dist(landmarks[self.L_CHEEK], landmarks[self.NOSE_TIP])
        r_dist = self._dist(landmarks[self.R_CHEEK], landmarks[self.NOSE_TIP])
        
        face_width = self._dist(landmarks[self.L_CHEEK], landmarks[self.R_CHEEK])
        diff = abs(l_dist - r_dist)
        
        # Normalization: difference of 10% face width is significant asymmetry
        score = diff / (face_width * 0.1)
        return float(np.clip(score, 0, 1))

    def get_features(self, current_landmarks, previous_landmarks=None):
        """Aggregates all computed features into a dictionary."""
        if current_landmarks is None:
            return None
            
        return {
            "eyebrow_raise": self.eyebrow_raise(current_landmarks),
            "lip_tension": self.lip_tension(current_landmarks),
            "blink_intensity": self.blink_rate(current_landmarks),
            "head_nod": self.head_nod_intensity(current_landmarks, previous_landmarks),
            "symmetry_delta": self.symmetry_delta(current_landmarks)
        }

if __name__ == "__main__":
    # Test with dummy data
    fe = FeatureEngineer()
    dummy_landmarks = np.random.rand(478, 3)
    features = fe.get_features(dummy_landmarks)
    print("Extracted Features:", features)
