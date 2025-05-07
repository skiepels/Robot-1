"""
Logger Module

This module implements a logging system for the trading strategy
to track strategy execution, trades, and performance.
"""

import logging
import os
import sys
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler

# Custom log formatter with colors for console output
class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console logs to improve readability.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Save original format
        orig_format = self._style._fmt
        
        # Apply the color for the level name
        self._style._fmt = (f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}"
                          f"{orig_format}{self.COLORS['RESET']}")
        
        # Call the original formatter
        result = logging.Formatter.format(self, record)
        
        # Restore the original format
        self._style._fmt = orig_format
        
        return result


# Formatter for file logs (no colors, detailed)
class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging to files.
    """
    def format(self, record):
        log_record = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Include exception info if available
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Include extra attributes
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 
                         'filename', 'funcName', 'id', 'levelname', 'levelno',
                         'lineno', 'module', 'msecs', 'message', 'msg', 'name', 
                         'pathname', 'process', 'processName', 'relativeCreated',
                         'stack_info', 'thread', 'threadName']:
                log_record[key] = value
        
        return json.dumps(log_record)


def setup_logger(name='trading_strategy', log_dir='logs', 
                console_level=logging.INFO, file_level=logging.DEBUG,
                max_file_size=10*1024*1024, backup_count=5):
    """
    Set up and configure a logger with console and file handlers.
    
    Parameters:
    -----------
    name: str
        Logger name
    log_dir: str
        Directory for log files
    console_level: int
        Logging level for console output
    file_level: int
        Logging level for file output
    max_file_size: int
        Maximum size of log file before rotation (in bytes)
    backup_count: int
        Number of backup log files to keep
        
    Returns:
    --------
    logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Create colored formatter for console
    console_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    console_formatter = ColoredFormatter(console_format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # Create file handler for detailed logging
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_file_size, backupCount=backup_count
    )
    file_handler.setLevel(file_level)
    
    # Create JSON formatter for file
    file_formatter = JsonFormatter()
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger