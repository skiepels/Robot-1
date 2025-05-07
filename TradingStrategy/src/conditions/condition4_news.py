"""
News Condition Module

Implements the fourth trading condition: stock has breaking news.
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsCondition:
    """
    Checks if a stock has breaking news.
    
    This condition helps identify stocks with catalysts that may drive significant price movement.
    News catalysts often generate increased interest and volume in a stock.
    """
    
    def __init__(self, max_hours_old=24):
        """
        Initialize the news condition.
        
        Parameters:
        -----------
        max_hours_old: int
            Maximum age of news in hours to be considered breaking
        """
        self.max_hours_old = max_hours_old
    
    def check(self, has_news=False, news_timestamp=None):
        """
        Check if the stock has breaking news.
        
        Parameters:
        -----------
        has_news: bool
            Whether the stock has associated news
        news_timestamp: datetime, optional
            Timestamp of the news
            
        Returns:
        --------
        bool: True if the stock has breaking news within the time window
        """
        if not has_news:
            return False
            
        # If no timestamp provided but has_news is True, assume it's breaking news
        if news_timestamp is None:
            return True
            
        # Check if news is recent enough
        now = datetime.now()
        max_age = timedelta(hours=self.max_hours_old)
        return (now - news_timestamp) <= max_age
    
    def check_by_headlines(self, headlines=None):
        """
        Check if any of the provided news headlines are significant.
        
        Parameters:
        -----------
        headlines: list of dict, optional
            List of news headline objects with 'timestamp' and 'headline' keys
            
        Returns:
        --------
        bool: True if significant breaking news is found
        """
        if not headlines:
            return False
            
        now = datetime.now()
        max_age = timedelta(hours=self.max_hours_old)
        
        # Keywords that often indicate significant news
        significant_keywords = [
            'announces', 'reports', 'launched', 'approved', 'partnership',
            'contract', 'agreement', 'merger', 'acquisition', 'acquires',
            'patent', 'breakthrough', 'fda', 'earnings', 'beats', 'exceeds',
            'raises', 'guidance', 'update', 'clinical', 'trial', 'results'
        ]
        
        for headline_obj in headlines:
            # Check if the headline is recent
            timestamp = headline_obj.get('timestamp')
            if timestamp and (now - timestamp) > max_age:
                continue
                
            # Check if the headline contains significant keywords
            headline = headline_obj.get('headline', '').lower()
            if any(keyword in headline for keyword in significant_keywords):
                logger.info(f"Significant news found: {headline}")
                return True
                
        return False
    
    def get_description(self):
        """Get a description of this condition."""
        return f"Has breaking news within the last {self.max_hours_old} hours"