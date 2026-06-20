class MEAError(Exception):
    """Base exception class for the Micro-Expression Analyzer."""
    pass

class CameraNotFoundError(MEAError):
    """Raised when a camera/video capture source cannot be opened."""
    pass

class ModelLoadError(MEAError):
    """Raised when a pre-trained ML model or landmark weights fail to load."""
    pass

class InvalidFrameError(MEAError):
    """Raised when a frame from the camera is empty, invalid, or corrupted."""
    pass

class ConfigurationError(MEAError):
    """Raised when configuration variables are invalid or missing."""
    pass

class LandmarkDetectionError(MEAError):
    """Raised when MediaPipe face landmarker fails to process a frame."""
    pass
