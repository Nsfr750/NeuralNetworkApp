"""
Logging configuration and utilities for the Neural Network application.
"""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = 'neuralnetwork', log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with daily rotating file and console handlers.
    Creates a new log file every day with the date in the filename.
    
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
        # File handler (daily rotation with date in filename)
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file = logs_dir / f'neuralnetwork_{current_date}.log'
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',  # Rotate at midnight
            interval=1,       # Every day
            backupCount=30,   # Keep 30 days of logs
            encoding='utf-8'
        )
        # Set the filename suffix for rotated files
        file_handler.suffix = '%Y-%m-%d'
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
