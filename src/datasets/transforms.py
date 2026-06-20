from torchvision import transforms
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("transforms")

def get_train_transforms(image_size=224):
    """
    Returns transforms for training with data augmentation.
    Loads values dynamically from config.yaml/default.yaml configuration.
    """
    # Default values
    flip = True
    degrees = 10
    brightness = 0.2
    contrast = 0.2
    saturation = 0.2
    
    try:
        config = load_config()
        aug_config = config.get("training", {}).get("augmentation", {})
        flip = aug_config.get("horizontal_flip", flip)
        degrees = aug_config.get("rotation_degree", degrees)
        brightness = aug_config.get("brightness_jitter", brightness)
        contrast = aug_config.get("contrast_jitter", contrast)
        saturation = aug_config.get("saturation_jitter", saturation)
    except Exception as e:
        logger.warning(f"Could not load config in get_train_transforms, using defaults: {e}")
        
    transform_list = [transforms.Resize((image_size, image_size))]
    
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees))
    if brightness > 0 or contrast > 0 or saturation > 0:
        transform_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        ))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def get_val_test_transforms(image_size=224):
    """
    Returns basic transforms for validation and testing.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
