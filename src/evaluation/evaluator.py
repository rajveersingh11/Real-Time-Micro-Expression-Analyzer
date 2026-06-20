import torch
import os
import json
from tqdm import tqdm

from src.datasets.dataloader_builder import build_dataloaders
from src.models.model_factory import build_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix_plot import plot_confusion_matrix
from src.evaluation.classification_report import generate_classification_report
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("evaluator")

def main():
    # Configuration
    MODEL_PATH = "models/best_model.pt"
    BATCH_SIZE = 64
    IMAGE_SIZE = 224
    NUM_CLASSES = 7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Label names in order of their mapped indices (0-6)
    labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    
    # 1. Load Data
    logger.info("Loading test data...")
    _, _, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0 # Set to 0 for stability in some environments
    )
    
    # 2. Build and Load Model
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, device=DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        logger.info("Model state_dict loaded successfully.")
    else:
        logger.warning(f"Model file {MODEL_PATH} not found. Using randomly initialized model for evaluation.")
    
    model.eval()
    
    # 3. Inference
    all_preds = []
    all_labels = []
    
    logger.info("Running inference on test set...")
    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader, desc="Evaluation"):
            images = images.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    # 4. Compute Metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(all_labels, all_preds)
    logger.info(f"Metrics: {metrics}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(LOG_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # 5. Generate Confusion Matrix
    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, labels, output_path=os.path.join(LOG_DIR, "confusion_matrix.png"))
    
    # 6. Generate Classification Report
    logger.info("Generating classification report...")
    generate_classification_report(all_labels, all_preds, labels, output_path=os.path.join(LOG_DIR, "classification_report.txt"))
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
