import torch
import os
import json
import sys
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.dataloader_builder import build_dataloaders
from src.models.model_factory import build_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix_plot import plot_confusion_matrix
from src.evaluation.classification_report import generate_classification_report

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
    print("Loading test data...")
    _, _, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0 # Set to 0 for stability in some environments
    )
    
    # 2. Build and Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, device=DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model state_dict loaded successfully.")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Using randomly initialized model for evaluation.")
    
    model.eval()
    
    # 3. Inference
    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader, desc="Evaluation"):
            images = images.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    # 4. Compute Metrics
    print("Computing metrics...")
    metrics = compute_metrics(all_labels, all_preds)
    print(f"Metrics: {metrics}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(LOG_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # 5. Generate Confusion Matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, labels, output_path=os.path.join(LOG_DIR, "confusion_matrix.png"))
    
    # 6. Generate Classification Report
    print("Generating classification report...")
    generate_classification_report(all_labels, all_preds, labels, output_path=os.path.join(LOG_DIR, "classification_report.txt"))
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
