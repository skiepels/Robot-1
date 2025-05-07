"""
Volume Condition Module

Implements the third trading condition: relative volume ratio of at least 5x.
"""

class VolumeCondition:
    """
    Checks if a stock has sufficient relative volume.
    
    This condition helps identify stocks with unusually high trading activity,
    which often indicates strong interest and potentially significant price movement.
    """
    
    def __init__(self, min_rel_volume=5.0):
        """
        Initialize the volume condition.
        
        Parameters:
        -----------
        min_rel_volume: float
            Minimum relative volume ratio (compared to average volume)
        """
        self.min_rel_volume = min_rel_volume
    
    def check(self, relative_volume):
        """
        Check if the relative volume meets the condition.
        
        Parameters:
        -----------
        relative_volume: float
            Current relative volume ratio
            
        Returns:
        --------
        bool: True if the relative volume is at least the minimum
        """
        if relative_volume is None:
            return False
            
        return relative_volume >= self.min_rel_volume
    
    def check_by_comparison(self, current_volume, average_volume):
        """
        Check volume condition by comparing current and average volume.
        
        Parameters:
        -----------
        current_volume: float
            Current trading volume
        average_volume: float
            Average trading volume (typically 50-day average)
            
        Returns:
        --------
        bool: True if the relative volume is at least the minimum
        """
        if current_volume is None or average_volume is None or average_volume == 0:
            return False
            
        relative_volume = current_volume / average_volume
        return relative_volume >= self.min_rel_volume
    
    def get_description(self):
        """Get a description of this condition."""
        return f"Relative volume at least {self.min_rel_volume:.1f}x average"