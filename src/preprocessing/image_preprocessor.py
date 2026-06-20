import cv2
import os
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("image_preprocessor")

def preprocess_image(input_path, output_path, size=(224, 224)):
    """
    Load image, resize directly, and save it.
    Avoids redundant color conversions.
    """
    try:
        # Load image (OpenCV loads as BGR by default)
        img = cv2.imread(input_path)
        if img is None:
            logger.error(f"Error loading image: {input_path}")
            return False
            
        # Resize image in BGR directly
        img_resized = cv2.resize(img, size)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed image
        cv2.imwrite(output_path, img_resized)
        
        return True
    except Exception as e:
        logger.error(f"Exception while preprocessing image {input_path}: {e}")
        return False
