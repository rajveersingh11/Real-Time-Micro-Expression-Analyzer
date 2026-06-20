import os
import zipfile
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("download_data")

def download_datasets(config_path="config/config.yaml"):
    """
    Downloads and extracts datasets according to config/config.yaml.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "data/artifacts")
    datasets = config.get("datasets", [])

    if not datasets:
        logger.warning("No datasets listed in configuration.")
        return

    # Initialize Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"Failed to authenticate Kaggle API. Please ensure kaggle.json is present: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        dataset_id = dataset.get("id")
        target_name = dataset.get("target_name")
        if not dataset_id or not target_name:
            continue

        logger.info(f"Downloading {dataset_id}...")
        
        try:
            api.dataset_download_files(dataset_id, path=output_dir, unzip=False)
            
            zip_filename = dataset_id.split("/")[-1] + ".zip"
            zip_path = os.path.join(output_dir, zip_filename)
            
            if os.path.exists(zip_path):
                logger.info(f"Unzipping {zip_filename}...")
                extract_to = os.path.join(output_dir, target_name)
                os.makedirs(extract_to, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                
                os.remove(zip_path)
                logger.info(f"Successfully downloaded and extracted to {extract_to}")
            else:
                logger.warning(f"{zip_filename} not found in {output_dir}. It might have been unzipped already.")
        except Exception as e:
            logger.error(f"Failed to download or extract {dataset_id}: {e}")

def main():
    download_datasets()

if __name__ == "__main__":
    main()
