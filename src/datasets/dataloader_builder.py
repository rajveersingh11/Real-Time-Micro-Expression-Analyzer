import os
import sys
from torch.utils.data import DataLoader
from .emotion_dataset import EmotionDataset
from .transforms import get_train_transforms, get_val_test_transforms

def build_dataloaders(
    train_csv="data/splits/train.csv",
    val_csv="data/splits/val.csv",
    test_csv="data/splits/test.csv",
    batch_size=64,
    num_workers=None,
    image_size=224
):
    """
    Builds and returns DataLoaders for train, val, and test sets.
    """
    if num_workers is None:
        num_workers = 0 if sys.platform.startswith('win') else 4
    
    # 1. Initialize Datasets with appropriate transforms
    train_dataset = EmotionDataset(
        csv_path=train_csv,
        transform=get_train_transforms(image_size=image_size)
    )
    
    val_dataset = EmotionDataset(
        csv_path=val_csv,
        transform=get_val_test_transforms(image_size=image_size)
    )
    
    test_dataset = EmotionDataset(
        csv_path=test_csv,
        transform=get_val_test_transforms(image_size=image_size)
    )
    
    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the loaders (run from project root: python -m src.datasets.dataloader_builder)
    train_loader, val_loader, test_loader = build_dataloaders(num_workers=0) # workers=0 for simple testing
    
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"First label: {labels[0]}")
        break
