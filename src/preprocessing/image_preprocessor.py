import cv2
import os

def preprocess_image(input_path, output_path, size=(224, 224)):
    """
    Load image, convert to RGB, resize to given size, and save it.
    """
    try:
        # Load image (OpenCV loads as BGR by default)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error loading image: {input_path}")
            return False
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, size)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed image (OpenCV expects BGR for imwrite, 
        # but since we converted to RGB for our internal standard, 
        # we'll save it as RGB by converting back to BGR for imwrite 
        # so it looks correct in most image viewers)
        img_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_to_save)
        
        return True
    except Exception as e:
        print(f"Exception while preprocessing image {input_path}: {e}")
        return False
