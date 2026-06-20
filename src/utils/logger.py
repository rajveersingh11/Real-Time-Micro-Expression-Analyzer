import os
import logging

def setup_logging(level="INFO"):
    """
    Configure logging format and handlers.
    Creates a 'logs' directory and configures stdout and file handlers.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    
    # Map level string to standard logging constants
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid double logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to initialize file logger at {log_file}: {e}")

def get_logger(name):
    """
    Convenience function to get a logger.
    """
    return logging.getLogger(name)
