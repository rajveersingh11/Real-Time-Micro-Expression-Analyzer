import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.dataloader_builder import build_dataloaders
from src.models.model_factory import build_model
from src.training.loss_functions import build_loss
from src.training.optimizer_builder import build_optimizer
from src.training.scheduler_builder import build_scheduler
from src.training.trainer import Trainer

def main():
    # Hyperparameters
    EPOCHS = 1
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 7
    IMAGE_SIZE = 224
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Build DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=4
    )
    
    # 2. Build Model
    model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, device=device)
    
    # 3. Build Training Components
    loss_fn = build_loss()
    optimizer = build_optimizer(model, lr=LEARNING_RATE)
    scheduler = build_scheduler(optimizer, epochs=EPOCHS)
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device
    )
    
    # 5. Start Training
    print("Starting training...")
    trainer.fit(epochs=EPOCHS)
    print("Training complete.")

if __name__ == "__main__":
    main()
