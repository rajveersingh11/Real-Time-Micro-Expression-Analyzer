from sklearn.metrics import classification_report
import os

def generate_classification_report(y_true, y_pred, labels, output_path="logs/classification_report.txt"):
    """
    Generates and saves a classification report.
    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        labels (list): List of label names.
        output_path (str): Path to save the report.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = classification_report(y_true, y_pred, target_names=labels)
    
    with open(output_path, "w") as f:
        f.write(report)
        
    print(f"Classification report saved to {output_path}")
    return report
