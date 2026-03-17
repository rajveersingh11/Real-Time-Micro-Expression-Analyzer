import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, labels, output_path="logs/confusion_matrix.png"):
    """
    Generates and saves a confusion matrix heatmap.
    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        labels (list): List of label names.
        output_path (str): Path to save the plot.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
