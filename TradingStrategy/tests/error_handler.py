"""
Error Handling Utility

This module provides enhanced error handling functionality for the trading strategy.
It includes decorators for exception handling, fallback mechanisms,
and improved logging for debugging.
"""

import os
import sys
import traceback
import logging
import time
import functools
import inspect
from datetime import datetime

# Get the root logger
logger = logging.getLogger('trading_strategy')


def exception_handler(func=None, retries=0, retry_delay=1, 
                    fallback_return=None, log_level=logging.ERROR,
                    notify=False):
    """
    Decorator for handling exceptions in functions.
    
    Parameters:
    -----------
    func: callable
        The function to decorate
    retries: int
        Number of times to retry the function on exception
    retry_delay: int
        Seconds to wait between retries
    fallback_return: any
        Value to return if all retries fail
    log_level: int
        Logging level for the exception
    notify: bool
        Whether to send a notification on exception
        
    Returns:
    --------
    callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function metadata for better logging
            module = error.get('module', 'Unknown')
            module_counts[module] = module_counts.get(module, 0) + 1
        
        # Generate the report
        report = "Error Analysis Report\n"
        report += "====================\n\n"
        
        report += f"Total Errors: {len(error_log)}\n\n"
        
        report += "Errors by Type:\n"
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"  {error_type}: {count}\n"
        
        report += "\nErrors by Severity:\n"
        for severity, count in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"  {severity}: {count}\n"
        
        report += "\nErrors by Module:\n"
        for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"  {module}: {count}\n"
        
        report += "\nMost Recent Errors:\n"
        recent_errors = sorted(error_log, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        
        for i, error in enumerate(recent_errors):
            report += f"\n{i+1}. {error.get('type', 'Unknown Error')} ({error.get('severity', 'ERROR')})\n"
            report += f"   Time: {error.get('timestamp', 'Unknown')}\n"
            report += f"   Module: {error.get('module', 'Unknown')}\n"
            report += f"   Function: {error.get('function', 'Unknown')}\n"
            report += f"   Message: {error.get('message', 'No message')}\n"
        
        # Write report to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


def send_error_notification(message, traceback=None):
    """
    Send an error notification.
    
    Parameters:
    -----------
    message: str
        Error message
    traceback: str, optional
        Error traceback
    """
    # This is a placeholder function that would integrate
    # with email, SMS, Slack, or other notification systems
    
    # For now, just log the error
    logger.error(f"NOTIFICATION: {message}")
    
    if traceback:
        logger.debug(f"Traceback: {traceback}")


# Example usage
if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('error_handler_test.log')
        ]
    )
    
    # Test the error handler
    @exception_handler(retries=2, retry_delay=1, fallback_return=None)
    def test_function(x, y):
        """Test function that divides x by y."""
        return x / y
    
    # Test with valid arguments
    result = test_function(10, 2)
    print(f"Result with valid args: {result}")
    
    # Test with invalid arguments
    result = test_function(10, 0)
    print(f"Result with invalid args: {result}")
    
    # Test the performance decorator
    @measure_performance(threshold_ms=100)
    def slow_function():
        """A deliberately slow function."""
        time.sleep(0.2)
        return "Done"
    
    slow_function()
    
    # Test the validation decorator
    def validate_positive_numbers(*args, **kwargs):
        """Validate that all arguments are positive numbers."""
        for arg in args:
            if not isinstance(arg, (int, float)) or arg <= 0:
                return False
        return True
    
    @validate_arguments(validate_positive_numbers)
    def calculate_area(width, height):
        """Calculate the area of a rectangle."""
        return width * height
    
    area1 = calculate_area(5, 10)
    print(f"Area with valid dimensions: {area1}")
    
    area2 = calculate_area(-5, 10)
    print(f"Area with invalid dimensions: {area2}")
    
    # Test the error handler class
    error_handler = ErrorHandler()
    
    try:
        result = 10 / 0
    except Exception as e:
        error_handler.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            module=__name__,
            function='<main>',
            severity='ERROR'
        )
    
    # Generate and print an error report
    report = error_handler.generate_error_report()
    print("\nError Report:")
    print(report)_name = func.__module__
            func_name = func.__name__
            
            # Get file name and line number
            try:
                filename = inspect.getfile(func)
                _, filename = os.path.split(filename)
                lineno = inspect.getsourcelines(func)[1]
            except:
                filename = "unknown"
                lineno = 0
            
            # Get caller info if available
            caller_info = ""
            current_frame = inspect.currentframe()
            if current_frame:
                caller_frame = current_frame.f_back
                if caller_frame:
                    caller_filename = caller_frame.f_code.co_filename
                    _, caller_filename = os.path.split(caller_filename)
                    caller_lineno = caller_frame.f_lineno
                    caller_func = caller_frame.f_code.co_name
                    caller_info = f", called from {caller_filename}:{caller_lineno} in {caller_func}()"
            
            # Try to execute the function with retries
            attempts = 0
            max_attempts = retries + 1
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    
                    # Format the exception message
                    exc_type = type(e).__name__
                    exc_msg = str(e)
                    
                    # Get the full traceback
                    tb = traceback.format_exc()
                    
                    # Create detailed error message
                    if attempts < max_attempts:
                        error_msg = (f"Exception in {module_name}.{func_name}() at {filename}:{lineno}{caller_info}, "
                                   f"attempt {attempts}/{max_attempts}: {exc_type}: {exc_msg}")
                    else:
                        error_msg = (f"Exception in {module_name}.{func_name}() at {filename}:{lineno}{caller_info}, "
                                   f"all {max_attempts} attempts failed: {exc_type}: {exc_msg}")
                    
                    # Log the error with appropriate level
                    if log_level == logging.DEBUG:
                        logger.debug(error_msg)
                        logger.debug(f"Traceback: {tb}")
                    elif log_level == logging.INFO:
                        logger.info(error_msg)
                        logger.debug(f"Traceback: {tb}")
                    elif log_level == logging.WARNING:
                        logger.warning(error_msg)
                        logger.debug(f"Traceback: {tb}")
                    elif log_level == logging.ERROR:
                        logger.error(error_msg)
                        logger.debug(f"Traceback: {tb}")
                    elif log_level == logging.CRITICAL:
                        logger.critical(error_msg)
                        logger.debug(f"Traceback: {tb}")
                    
                    # Send notification if requested
                    if notify:
                        send_error_notification(error_msg, tb)
                    
                    # Wait before retrying
                    if attempts < max_attempts:
                        time.sleep(retry_delay)
            
            # If we get here, all retries failed
            return fallback_return
        
        return wrapper
    
    # Handle case where decorator is used without arguments
    if func is not None:
        return decorator(func)
    
    return decorator


def timeout_handler(timeout=10, fallback_return=None):
    """
    Decorator for handling function timeouts.
    
    Parameters:
    -----------
    timeout: int
        Maximum seconds to allow the function to run
    fallback_return: any
        Value to return if the function times out
        
    Returns:
    --------
    callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Import here to avoid dependencies when not used
            import signal
            
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
            
            # Set the timeout
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            
            try:
                result = func(*args, **kwargs)
                # Cancel the alarm if the function succeeds
                signal.alarm(0)
                return result
            except TimeoutError as e:
                # Log the timeout
                logger.warning(f"Timeout in {func.__module__}.{func.__name__}(): {str(e)}")
                return fallback_return
            finally:
                # Ensure alarm is canceled
                signal.alarm(0)
        
        return wrapper
    
    return decorator


def validate_arguments(validator_func):
    """
    Decorator for validating function arguments.
    
    Parameters:
    -----------
    validator_func: callable
        Function that takes the same arguments as the decorated function
        and returns True if valid, False otherwise
        
    Returns:
    --------
    callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the validator function
            is_valid = validator_func(*args, **kwargs)
            
            if not is_valid:
                # Log the validation failure
                arg_values = ", ".join([str(arg) for arg in args])
                kwarg_values = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                call_sig = f"{arg_values}{', ' if arg_values and kwarg_values else ''}{kwarg_values}"
                
                logger.warning(f"Validation failed for {func.__module__}.{func.__name__}({call_sig})")
                
                # Return None for invalid arguments
                return None
            
            # Proceed with the function if validation passes
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def measure_performance(func=None, threshold_ms=None):
    """
    Decorator for measuring function performance.
    
    Parameters:
    -----------
    func: callable
        The function to decorate
    threshold_ms: int, optional
        Threshold in milliseconds, above which to log performance warnings
        
    Returns:
    --------
    callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function metadata
            module_name = func.__module__
            func_name = func.__name__
            
            # Start timing
            start_time = time.time()
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Measure execution time
            execution_time = time.time() - start_time
            execution_time_ms = execution_time * 1000
            
            # Log performance data
            if threshold_ms is not None and execution_time_ms > threshold_ms:
                logger.warning(f"Performance warning: {module_name}.{func_name}() "
                             f"took {execution_time_ms:.2f}ms (threshold: {threshold_ms}ms)")
            else:
                logger.debug(f"Performance: {module_name}.{func_name}() took {execution_time_ms:.2f}ms")
            
            return result
        
        return wrapper
    
    # Handle case where decorator is used without arguments
    if func is not None:
        return decorator(func)
    
    return decorator


class ErrorHandler:
    """
    Class for handling and logging errors centrally.
    """
    
    def __init__(self, log_dir='logs'):
        """
        Initialize the error handler.
        
        Parameters:
        -----------
        log_dir: str
            Directory for error logs
        """
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create error log file
        self.error_log_file = os.path.join(log_dir, 'error_log.json')
        
        # Initialize error log if it doesn't exist
        if not os.path.exists(self.error_log_file):
            with open(self.error_log_file, 'w') as f:
                f.write('[]')
    
    def log_error(self, error_type, error_message, traceback=None, 
                module=None, function=None, severity='ERROR'):
        """
        Log an error to the error log file.
        
        Parameters:
        -----------
        error_type: str
            Type of error
        error_message: str
            Error message
        traceback: str, optional
            Error traceback
        module: str, optional
            Module where the error occurred
        function: str, optional
            Function where the error occurred
        severity: str
            Error severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        import json
        
        # Create error entry
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'traceback': traceback,
            'module': module,
            'function': function,
            'severity': severity
        }
        
        # Read existing log
        try:
            with open(self.error_log_file, 'r') as f:
                error_log = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            error_log = []
        
        # Add new entry
        error_log.append(error_entry)
        
        # Write back to file
        with open(self.error_log_file, 'w') as f:
            json.dump(error_log, f, indent=2)
        
        # Log to logging system as well
        log_func = getattr(logger, severity.lower(), logger.error)
        log_func(f"{error_type} in {module}.{function}: {error_message}")
    
    def get_recent_errors(self, count=10, severity=None):
        """
        Get the most recent errors from the error log.
        
        Parameters:
        -----------
        count: int
            Number of errors to retrieve
        severity: str, optional
            Filter by severity
            
        Returns:
        --------
        list: Recent errors
        """
        import json
        
        # Read error log
        try:
            with open(self.error_log_file, 'r') as f:
                error_log = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        
        # Filter by severity if specified
        if severity:
            error_log = [e for e in error_log if e.get('severity') == severity]
        
        # Sort by timestamp (newest first)
        error_log.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Return the specified number of errors
        return error_log[:count]
    
    def generate_error_report(self, output_file=None):
        """
        Generate a report of all errors.
        
        Parameters:
        -----------
        output_file: str, optional
            Output file for the report
            
        Returns:
        --------
        str: Error report
        """
        import json
        
        # Read error log
        try:
            with open(self.error_log_file, 'r') as f:
                error_log = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            error_log = []
        
        # Count errors by type
        error_counts = {}
        for error in error_log:
            error_type = error.get('type', 'Unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Count errors by severity
        severity_counts = {}
        for error in error_log:
            severity = error.get('severity', 'ERROR')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count errors by module
        module_counts = {}
        for error in error_log:
            module