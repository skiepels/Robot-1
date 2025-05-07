"""
Price Condition Module

Implements the first trading condition: stock price between $2 and $20.
"""

class PriceCondition:
    """
    Checks if a stock's price is within the desired range.
    
    This condition ensures that we're trading stocks that have enough
    price movement potential while not being too expensive or too cheap.
    """
    
    def __init__(self, min_price=2.0, max_price=20.0):
        """
        Initialize the price condition.
        
        Parameters:
        -----------
        min_price: float
            Minimum acceptable stock price
        max_price: float
            Maximum acceptable stock price
        """
        self.min_price = min_price
        self.max_price = max_price
    
    def check(self, price):
        """
        Check if the price meets the condition.
        
        Parameters:
        -----------
        price: float
            Current stock price
            
        Returns:
        --------
        bool: True if price is within the acceptable range
        """
        if price is None:
            return False
            
        return self.min_price <= price <= self.max_price
    
    def get_description(self):
        """Get a description of this condition."""
        return f"Price between ${self.min_price:.2f} and ${self.max_price:.2f}"