"""
Logger Module

This module implements a logging system for the day trading strategy
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


# Trade logger for detailed trade information
class TradeLogger:
    """
    Specialized logger for trade execution and management.
    """
    def __init__(self, log_dir='logs', trade_log_file='trades.json'):
        self.log_dir = log_dir
        self.trade_log_file = os.path.join(log_dir, trade_log_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize empty trade log if file doesn't exist
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w') as f:
                json.dump([], f)
    
    def log_trade(self, trade_data):
        """
        Log trade data to JSON file.
        """
        # Read existing trades
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            trades = []
        
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Append new trade
        trades.append(trade_data)
        
        # Write back to file
        with open(self.trade_log_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def get_trades(self, limit=None, start_date=None, end_date=None, symbol=None):
        """
        Retrieve trades from the log file with optional filtering.
        
        Parameters:
        -----------
        limit: int, optional
            Maximum number of trades to return
        start_date: str, optional
            ISO format date string for filtering trades after this date
        end_date: str, optional
            ISO format date string for filtering trades before this date
        symbol: str, optional
            Filter trades by symbol
            
        Returns:
        --------
        list: Filtered trade records
        """
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        
        # Apply filters
        if start_date:
            start_datetime = datetime.fromisoformat(start_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= start_datetime]
        
        if end_date:
            end_datetime = datetime.fromisoformat(end_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) <= end_datetime]
        
        if symbol:
            trades = [t for t in trades if t.get('symbol') == symbol]
        
        # Apply limit
        if limit and limit > 0:
            trades = trades[-limit:]
        
        return trades


# Performance logger for strategy performance metrics
class PerformanceLogger:
    """
    Specialized logger for tracking trading performance metrics.
    """
    def __init__(self, log_dir='logs', performance_log_file='performance.json'):
        self.log_dir = log_dir
        self.performance_log_file = os.path.join(log_dir, performance_log_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_daily_performance(self, performance_data):
        """
        Log daily performance metrics.
        """
        # Read existing performance data
        try:
            with open(self.performance_log_file, 'r') as f:
                performance_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            performance_history = {}
        
        # Add timestamp if not present
        if 'date' not in performance_data:
            performance_data['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add or update performance for the day
        date_key = performance_data['date']
        performance_history[date_key] = performance_data
        
        # Write back to file
        with open(self.performance_log_file, 'w') as f:
            json.dump(performance_history, f, indent=2)
    
    def get_performance_history(self, days=30):
        """
        Retrieve performance history.
        
        Parameters:
        -----------
        days: int, optional
            Number of days to retrieve
            
        Returns:
        --------
        dict: Performance history
        """
        try:
            with open(self.performance_log_file, 'r') as f:
                performance_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
        
        # Sort by date and limit to requested number of days
        sorted_dates = sorted(performance_history.keys(), reverse=True)
        recent_dates = sorted_dates[:days]
        
        return {date: performance_history[date] for date in recent_dates}


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


# Create trade and performance loggers
def get_trade_logger(log_dir='logs'):
    """Get the trade logger instance."""
    return TradeLogger(log_dir=log_dir)


def get_performance_logger(log_dir='logs'):
    """Get the performance logger instance."""
    return PerformanceLogger(log_dir=log_dir)