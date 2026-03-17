import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading micro-expression images and labels.
    """
    
    LABEL_MAPPING = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "happiness": 3,
        "sadness": 4,
        "surprise": 5,
        "neutral": 6
    }

    def __init__(self, csv_path, transform=None):
        """
        Initializes the dataset.
        Args:
            csv_path (str): Path to the CSV file (train, val, or test).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Verify and clean the dataset (optional: remove missing files now or handle in __getitem__)
        self.valid_indices = self._get_valid_indices()
        print(f"Loaded {len(self.valid_indices)} valid samples from {csv_path} (out of {len(self.df)})")

    def _get_valid_indices(self):
        """
        Returns a list of indices where the image path exists.
        """
        valid_indices = []
        for idx, row in self.df.iterrows():
            if os.path.exists(row['image_path']):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map dataset index to internal valid index
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        img_path = row['image_path']
        label_str = row['label']
        label = self.LABEL_MAPPING[label_str]
        
        try:
            # Load image using Pillow and ensure RGB mode
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.long)
            
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # In a real training scenario, you'd want to handle this more gracefully
            # e.g., by returning a dummy image or choosing another index
            # For now, we'll return a zero tensor if it fails (not ideal)
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.long)
