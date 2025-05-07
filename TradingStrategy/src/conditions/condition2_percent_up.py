"""
Percent Up Condition Module

Implements the second trading condition: stock is up at least 10% on the day.
"""

class PercentUpCondition:
    """
    Checks if a stock is up by at least the minimum percentage on the day.
    
    This condition helps identify stocks with strong momentum and volatility.
    """
    
    def __init__(self, min_percent=10.0):
        """
        Initialize the percent up condition.
        
        Parameters:
        -----------
        min_percent: float
            Minimum percentage a stock should be up on the day
        """
        self.min_percent = min_percent
    
    def check(self, percent_change):
        """
        Check if the percentage change meets the condition.
        
        Parameters:
        -----------
        percent_change: float
            Current percentage change from previous close
            
        Returns:
        --------
        bool: True if the stock is up by at least the minimum percentage
        """
        if percent_change is None:
            return False
            
        return percent_change >= self.min_percent
    
    def get_description(self):
        """Get a description of this condition."""
        return f"Up at least {self.min_percent:.1f}% on the day"