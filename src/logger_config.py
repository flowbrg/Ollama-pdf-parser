import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_to_file: bool = True, log_dir: str = "./logs"):
    """Simple logging setup for the pipeline"""
    
    # Create log directory if needed
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"pipeline_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    return logging.getLogger("pipeline")

def get_logger(name: str = "pipeline"):
    """Get a logger instance"""
    return logging.getLogger(name)