"""
Float Condition Module

Implements the fifth trading condition: stock has a float of less than 20 million shares.
"""

class FloatCondition:
    """
    Checks if a stock has a sufficiently low float.
    
    This condition identifies stocks with limited supply, which can lead to
    greater imbalance between supply and demand, potentially creating
    more dramatic price movements.
    """
    
    def __init__(self, max_float=20_000_000, ideal_max=10_000_000):
        """
        Initialize the float condition.
        
        Parameters:
        -----------
        max_float: int
            Maximum acceptable float size in shares
        ideal_max: int
            Ideal maximum float size for best performance
        """
        self.max_float = max_float
        self.ideal_max = ideal_max
    
    def check(self, float_size):
        """
        Check if the float size meets the condition.
        
        Parameters:
        -----------
        float_size: int
            Float size in shares
            
        Returns:
        --------
        bool: True if the float is below the maximum
        """
        if float_size is None:
            return False
            
        return float_size <= self.max_float
    
    def quality_score(self, float_size):
        """
        Get a quality score (0-1) based on how good the float size is.
        
        Parameters:
        -----------
        float_size: int
            Float size in shares
            
        Returns:
        --------
        float: Quality score from 0 to 1 (higher is better)
        """
        if float_size is None or float_size > self.max_float:
            return 0.0
            
        if float_size <= 1_000_000:
            return 1.0  # Best possible float size
            
        if float_size <= 5_000_000:
            return 0.8  # Very good float size
            
        if float_size <= self.ideal_max:
            return 0.6  # Good float size
            
        # Linear scale between ideal_max and max_float
        score = 0.6 * (self.max_float - float_size) / (self.max_float - self.ideal_max)
        return max(0.0, min(score, 0.6))
    
    def get_description(self):
        """Get a description of this condition."""
        return f"Float less than {self.max_float:,} shares (ideally less than {self.ideal_max:,})"