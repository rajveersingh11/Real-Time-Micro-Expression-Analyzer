import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from label_mapper import map_label
from image_preprocessor import preprocess_image

def process_fer2013(raw_dir, processed_images_dir):
    data = []
    # User asked to iterate through folders
    for split in ["train", "test"]:
        split_dir = os.path.join(raw_dir, "fer2013", split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"Processing FER2013 {split} split...")
        for label_folder in os.listdir(split_dir):
            unified_label = map_label("fer2013", label_folder)
            if not unified_label:
                continue
                
            folder_path = os.path.join(split_dir, label_folder)
            for img_name in tqdm(os.listdir(folder_path), desc=f"FER {label_folder}"):
                img_path = os.path.join(folder_path, img_name)
                output_name = f"fer_{split}_{label_folder}_{img_name}"
                output_path = os.path.join(processed_images_dir, output_name)
                
                if preprocess_image(img_path, output_path):
                    data.append({
                        "image_path": output_path,
                        "label": unified_label,
                        "dataset_source": "FER"
                    })
    return data

def process_raf_db(raw_dir, processed_images_dir):
    data = []
    raf_dir = os.path.join(raw_dir, "raf-db")
    
    for split in ["train", "test"]:
        csv_path = os.path.join(raf_dir, f"{split}_labels.csv")
        if not os.path.exists(csv_path):
            continue
            
        print(f"Processing RAF-DB {split} split...")
        df = pd.read_csv(csv_path)
        img_dir = os.path.join(raf_dir, "DATASET", split)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RAF {split}"):
            raw_label = row["label"]
            unified_label = map_label("raf-db", raw_label)
            if not unified_label:
                continue
                
            img_name = row["image"]
            # RAF images are stored in subfolders named after their labels (1-7)
            img_path = os.path.join(img_dir, str(raw_label), img_name)
            output_name = f"raf_{split}_{img_name}"
            output_path = os.path.join(processed_images_dir, output_name)
            
            if preprocess_image(img_path, output_path):
                data.append({
                    "image_path": output_path,
                    "label": unified_label,
                    "dataset_source": "RAF"
                })
    return data

def process_ckplus(raw_dir, processed_images_dir):
    data = []
    csv_path = os.path.join(raw_dir, "ckplus", "ckextended.csv")
    if not os.path.exists(csv_path):
        return data
        
    print("Processing CK+ (from CSV pixels)...")
    df = pd.read_csv(csv_path)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="CK+"):
        unified_label = map_label("ckplus", row["emotion"])
        if not unified_label:
            continue
            
        # Pixels are space separated string
        pixels = np.fromstring(row["pixels"], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        
        output_name = f"ck_{i}.jpg"
        output_path = os.path.join(processed_images_dir, output_name)
        
        # Save the pixel data as image first, then use our preprocessor (or just save directly)
        # Since pixels are already gray, we convert to RGB for consistency
        img_rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        
        data.append({
            "image_path": output_path,
            "label": unified_label,
            "dataset_source": "CK+"
        })
    return data

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    images_dir = os.path.join(processed_dir, "images")
    splits_dir = "data/splits"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    all_data = []
    
    # Process each dataset
    all_data.extend(process_fer2013(raw_dir, images_dir))
    all_data.extend(process_raf_db(raw_dir, images_dir))
    all_data.extend(process_ckplus(raw_dir, images_dir))
    
    if not all_data:
        print("No data processed. Please check raw data directories.")
        return
        
    full_df = pd.DataFrame(all_data)
    
    # Save unified metadata
    full_df.to_csv(os.path.join(processed_dir, "unified_metadata.csv"), index=False)
    print(f"Total images processed: {len(full_df)}")
    
    # Create splits (70/15/15)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.3, random_state=42, stratify=full_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )
    
    # Save splits
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    
    print(f"Splits created: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    print("Preprocessing pipeline complete.")

if __name__ == "__main__":
    main()
