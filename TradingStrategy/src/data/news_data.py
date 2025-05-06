"""
News Data Provider

This module handles fetching and processing news data for the day trading strategy.
It implements methods to get stock news and identify catalysts that could drive price movement,
which is a key factor in Ross Cameron's strategy.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import time
import re

logger = logging.getLogger(__name__)


class NewsItem:
    def __init__(self, headline, source, url, date=None, summary=None):
        """
        Initialize a news item object.
        
        Parameters:
        -----------
        headline: str
            The news headline
        source: str
            The source of the news (e.g., "Bloomberg", "CNBC")
        url: str
            URL to the full news article
        date: datetime, optional
            Publication date and time
        summary: str, optional
            Brief summary of the news
        """
        self.headline = headline
        self.source = source
        self.url = url
        self.date = date or datetime.now()
        self.summary = summary
        self.score = 0  # Impact score (to be calculated)
    
    def __str__(self):
        """String representation of the news item"""
        return f"{self.date.strftime('%Y-%m-%d %H:%M')} - {self.headline} ({self.source})"


class NewsDataProvider:
    def __init__(self, api_key=None):
        """
        Initialize the news data provider.
        
        Parameters:
        -----------
        api_key: str, optional
            API key for news data services (if used)
        """
        self.api_key = api_key
        self.cached_news = {}  # Cache for news data
        self.cache_expiry = {}  # Expiry timestamps for cached news
        
        # Keywords that indicate potentially significant news
        self.catalyst_keywords = {
            'high_impact': [
                'fda approval', 'breakthrough', 'patent', 'merger', 
                'acquisition', 'buyout', 'takeover', 'announces',
                'contract', 'partnership', 'deal', 'secures'
            ],
            'medium_impact': [
                'earnings', 'beats', 'misses', 'revenue', 'guidance',
                'outlook', 'forecast', 'raises', 'lowers', 'dividend',
                'launch', 'release', 'unveils', 'introduces'
            ],
            'low_impact': [
                'appoints', 'names', 'hires', 'joins', 'leaves',
                'conference', 'presents', 'speaking', 'interview',
                'analyst', 'rating', 'upgrade', 'downgrade'
            ]
        }
    
    def get_stock_news(self, symbol, days=1, max_items=10):
        """
        Get recent news for a specific stock.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
        days: int
            Number of days to look back
        max_items: int
            Maximum number of news items to return
            
        Returns:
        --------
        list: NewsItem objects for the specified stock
        """
        logger.info(f"Fetching news for {symbol} for the past {days} days...")
        
        # Check cache first
        cache_key = f'news_{symbol}_{days}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached news data")
            return self.cached_news[cache_key]
        
        try:
            # In production, this would connect to a news API
            # For demonstration, we'll use a simulated news feed
            news_items = self._simulate_news_feed(symbol, days)
            
            # Sort by date (newest first)
            news_items.sort(key=lambda x: x.date, reverse=True)
            
            # Limit to the requested number of items
            if max_items > 0:
                news_items = news_items[:max_items]
            
            # Score each news item for potential impact
            for item in news_items:
                item.score = self._calculate_news_impact_score(item)
            
            # Cache the results
            self._cache_data(cache_key, news_items, expiry_seconds=300)  # 5 min expiry
            
            logger.info(f"Fetched {len(news_items)} news items for {symbol}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_market_news(self, days=1, max_items=20):
        """
        Get recent general market news.
        
        Parameters:
        -----------
        days: int
            Number of days to look back
        max_items: int
            Maximum number of news items to return
            
        Returns:
        --------
        list: NewsItem objects for general market news
        """
        logger.info(f"Fetching general market news for the past {days} days...")
        
        # Check cache first
        cache_key = f'market_news_{days}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached market news data")
            return self.cached_news[cache_key]
        
        try:
            # In production, this would connect to a news API
            # For demonstration, we'll use a simulated news feed
            news_items = self._simulate_market_news_feed(days)
            
            # Sort by date (newest first)
            news_items.sort(key=lambda x: x.date, reverse=True)
            
            # Limit to the requested number of items
            if max_items > 0:
                news_items = news_items[:max_items]
            
            # Cache the results
            self._cache_data(cache_key, news_items, expiry_seconds=900)  # 15 min expiry
            
            logger.info(f"Fetched {len(news_items)} general market news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def find_stocks_with_catalysts(self, symbols, min_impact_score=5):
        """
        Find stocks from a list that have significant news catalysts.
        
        Parameters:
        -----------
        symbols: list
            List of stock ticker symbols to check
        min_impact_score: int
            Minimum news impact score to consider significant
            
        Returns:
        --------
        dict: Dictionary with symbols as keys and their most significant news as values
        """
        logger.info(f"Finding news catalysts for {len(symbols)} stocks...")
        
        stocks_with_catalysts = {}
        
        for symbol in symbols:
            # Get recent news for this stock
            news_items = self.get_stock_news(symbol, days=1)
            
            # Find the news item with the highest impact score
            if news_items:
                highest_impact_news = max(news_items, key=lambda x: x.score)
                
                # If the score meets the threshold, add to results
                if highest_impact_news.score >= min_impact_score:
                    stocks_with_catalysts[symbol] = highest_impact_news
        
        logger.info(f"Found {len(stocks_with_catalysts)} stocks with significant news catalysts")
        return stocks_with_catalysts
    
    def _simulate_news_feed(self, symbol, days):
        """
        Simulate a news feed for a given stock.
        This is used for demonstration when a real news API is not available.
        """
        # Generate some plausible news items
        news_templates = [
            "{symbol} Announces Quarterly Earnings",
            "{symbol} Secures New Contract",
            "{symbol} Receives FDA Approval for Key Product",
            "{symbol} Expands into New Markets",
            "{symbol} Partners with Major Industry Player",
            "{symbol} Announces Stock Buyback Program",
            "{symbol} Raises Guidance for Upcoming Quarter",
            "{symbol} CEO Featured in Industry Interview",
            "{symbol} Introduces New Product Line",
            "{symbol} Reports Record Sales"
        ]
        
        sources = ["Reuters", "Bloomberg", "CNBC", "Yahoo Finance", "MarketWatch"]
        
        # Generate a random number of news items (0-5)
        import random
        num_items = random.randint(0, 5)
        
        news_items = []
        now = datetime.now()
        
        for i in range(num_items):
            # Pick a random template and source
            template = random.choice(news_templates)
            source = random.choice(sources)
            
            # Generate a headline
            headline = template.format(symbol=symbol)
            
            # Generate a random date within the specified days
            random_hours = random.randint(0, days * 24)
            date = now - timedelta(hours=random_hours)
            
            # Create a dummy URL
            url = f"https://example.com/news/{symbol.lower()}/{i}"
            
            # Create the news item
            news_item = NewsItem(headline, source, url, date)
            news_items.append(news_item)
        
        return news_items
    
    def _simulate_market_news_feed(self, days):
        """
        Simulate a general market news feed.
        This is used for demonstration when a real news API is not available.
        """
        # Generate some plausible market news items
        news_templates = [
            "Markets React to Fed Decision on Interest Rates",
            "S&P 500 Reaches New All-Time High",
            "Tech Stocks Lead Market Rally",
            "Inflation Data Impacts Market Sentiment",
            "Global Economic Outlook Affects Trading",
            "Market Volatility Increases as Earnings Season Begins",
            "Investors Respond to Latest Economic Indicators",
            "Market Trends: Sector Rotation Observed",
            "Futures Point to Mixed Open After Previous Session",
            "Treasury Yields Affect Market Dynamics"
        ]
        
        sources = ["Reuters", "Bloomberg", "CNBC", "Yahoo Finance", "MarketWatch"]
        
        # Generate a random number of news items (5-10)
        import random
        num_items = random.randint(5, 10)
        
        news_items = []
        now = datetime.now()
        
        for i in range(num_items):
            # Pick a random template and source
            template = random.choice(news_templates)
            source = random.choice(sources)
            
            # Generate a headline
            headline = template
            
            # Generate a random date within the specified days
            random_hours = random.randint(0, days * 24)
            date = now - timedelta(hours=random_hours)
            
            # Create a dummy URL
            url = f"https://example.com/market-news/{i}"
            
            # Create the news item
            news_item = NewsItem(headline, source, url, date)
            news_items.append(news_item)
        
        return news_items
    
    def _calculate_news_impact_score(self, news_item):
        """
        Calculate a potential impact score for a news item.
        This helps identify which news items are likely to be market-moving catalysts.
        
        Higher scores indicate potentially higher impact.
        """
        headline = news_item.headline.lower()
        
        # Start with a base score
        score = 0
        
        # Check for high impact keywords
        for keyword in self.catalyst_keywords['high_impact']:
            if keyword in headline:
                score += 3
        
        # Check for medium impact keywords
        for keyword in self.catalyst_keywords['medium_impact']:
            if keyword in headline:
                score += 2
        
        # Check for low impact keywords
        for keyword in self.catalyst_keywords['low_impact']:
            if keyword in headline:
                score += 1
        
        # Recency bonus: news within the last 3 hours gets extra points
        hours_ago = (datetime.now() - news_item.date).total_seconds() / 3600
        if hours_ago <= 3:
            score += 3
        elif hours_ago <= 6:
            score += 2
        elif hours_ago <= 12:
            score += 1
        
        # Source credibility bonus
        credible_sources = ["Bloomberg", "Reuters", "CNBC", "Wall Street Journal", "Financial Times"]
        if any(source.lower() in news_item.source.lower() for source in credible_sources):
            score += 1
        
        return score
    
    def _cache_data(self, key, data, expiry_seconds=300):
        """Cache data with an expiry time"""
        self.cached_news[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=expiry_seconds)
    
    def _is_cache_valid(self, key):
        """Check if cached data exists and is still valid"""
        if key not in self.cached_news or key not in self.cache_expiry:
            return False
            
        return datetime.now() < self.cache_expiry[key]