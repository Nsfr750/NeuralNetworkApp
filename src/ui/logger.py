"""
Logging configuration and utilities for the Neural Network application.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str = 'neuralnetwork', log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    try:
        # File handler (rotating)
        log_file = logs_dir / 'neuralnetwork.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create file logger: {e}", file=sys.stderr)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Optional logger name. If None, returns the root logger.
        
    Returns:
        Configured logger instance
    """
    if name:
        return logging.getLogger(f'neuralnetwork.{name}')
    return logger
