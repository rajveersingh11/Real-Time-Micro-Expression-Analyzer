import torch
import os
import random
import numpy as np

from src.datasets.dataloader_builder import build_dataloaders
from src.models.model_factory import build_model
from src.training.loss_functions import build_loss
from src.training.optimizer_builder import build_optimizer
from src.training.scheduler_builder import build_scheduler
from src.training.trainer import Trainer
from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config

setup_logging()
logger = get_logger("train")

def main():
    # 1. Load configuration
    try:
        config = load_config()
        epochs = config["training"].get("epochs", 50)
        batch_size = config["training"].get("batch_size", 64)
        learning_rate = config["training"].get("learning_rate", 0.001)
        seed = config["training"].get("seed", 42)
    except Exception as e:
        logger.warning(f"Failed to load config, using default settings: {e}")
        epochs = 50
        batch_size = 64
        learning_rate = 0.001
        seed = 42

    num_classes = config.get("model", {}).get("num_classes", 7)
    image_size = config.get("model", {}).get("image_size", 224)

    # Set reproducibility seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} for reproducibility.")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Build DataLoaders
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        image_size=image_size
    )
    
    # 3. Build Model
    model = build_model(model_name="resnet18", num_classes=num_classes, device=device)
    
    # 4. Build Training Components
    loss_fn = build_loss()
    optimizer = build_optimizer(model, lr=learning_rate)
    scheduler = build_scheduler(optimizer, epochs=epochs)
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device
    )
    
    # 6. Start Training
    logger.info("Starting training...")
    trainer.fit(epochs=epochs)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
