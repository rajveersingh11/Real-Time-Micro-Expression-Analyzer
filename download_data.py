import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_datasets():
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define datasets and their target names (for unzipping)
    datasets = {
        "msambare/fer2013": "fer2013",
        "shuvoalok/raf-db-dataset": "raf-db",
        "davilsena/ckdataset": "ckplus"
    }

    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    for dataset_id, target_name in datasets.items():
        print(f"Downloading {dataset_id}...")
        
        # Download the dataset
        api.dataset_download_files(dataset_id, path=raw_data_dir, unzip=False)
        
        # Find the downloaded zip file
        zip_filename = dataset_id.split("/")[-1] + ".zip"
        zip_path = os.path.join(raw_data_dir, zip_filename)
        
        # Check if the zip file exists (sometimes Kaggle API unzips it if specified, 
        # but we'll manually ensure it's extracted to a clean directory)
        if os.path.exists(zip_path):
            print(f"Unzipping {zip_filename}...")
            extract_to = os.path.join(raw_data_dir, target_name)
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # Clean up the zip file
            os.remove(zip_path)
            print(f"Successfully downloaded and extracted to {extract_to}")
        else:
            print(f"Warning: {zip_filename} not found in {raw_data_dir}. It might have been unzipped already.")

if __name__ == "__main__":
    download_datasets()
    print("All datasets downloaded and processed.")
