"""
Mock Data Generator

This module provides mock data for backtesting the trading strategy.
It generates synthetic price and news data that can be used for testing
without requiring actual market data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class MockDataGenerator:
    """Generator for mock market and news data for backtesting."""
    
    def __init__(self, start_date, end_date):
        """
        Initialize the mock data generator.
        
        Parameters:
        -----------
        start_date: datetime
            Start date for data generation
        end_date: datetime
            End date for data generation
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # Sample stock universe for backtesting
        self.stock_universe = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD',
            'INTC', 'NFLX', 'PYPL', 'SQ', 'TWTR', 'SNAP', 'UBER', 'LYFT',
            'ZM', 'SHOP', 'ROKU', 'ETSY', 'BABA', 'NIO', 'PLTR', 'COIN',
            'GME', 'AMC', 'BB', 'NOK', 'SPCE', 'TLRY'
        ]
        
        # Generated data storage
        self.price_data = {}
        self.news_data = {}
    
    def generate_all_data(self):
        """Generate price and news data for all stocks in the universe."""
        self.generate_price_data()
        self.generate_news_data()
    
    def generate_price_data(self):
        """Generate synthetic price data for backtesting."""
        logger.info("Generating mock price data...")
        
        total_stocks = len(self.stock_universe)
        
        for i, symbol in enumerate(self.stock_universe):
            logger.info(f"Generating data for {symbol} ({i+1}/{total_stocks})...")
            
            # Generate daily data for each stock
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            
            # Skip weekends
            dates = [date for date in dates if date.weekday() < 5]
            
            # Base price and daily volatility
            base_price = np.random.uniform(5, 100)
            volatility = np.random.uniform(0.01, 0.05)
            
            # Generate daily prices with random walk
            daily_returns = np.random.normal(0.0005, volatility, size=len(dates))
            daily_prices = base_price * (1 + np.cumsum(daily_returns))
            
            # Create DataFrame with daily prices
            daily_data = pd.DataFrame({
                'date': dates,
                'open': daily_prices * np.random.uniform(0.99, 1.01, size=len(dates)),
                'high': daily_prices * np.random.uniform(1.01, 1.05, size=len(dates)),
                'low': daily_prices * np.random.uniform(0.95, 0.99, size=len(dates)),
                'close': daily_prices,
                'volume': np.random.randint(100000, 10000000, size=len(dates))
            })
            
            # Generate intraday data for each day
            intraday_data = {}
            
            for date in dates:
                # Trading hours (9:30 AM to 4:00 PM)
                times = pd.date_range(
                    start=date.replace(hour=9, minute=30),
                    end=date.replace(hour=16, minute=0),
                    freq='1min'
                )
                
                # Get daily data for this date
                day_data = daily_data[daily_data['date'] == date].iloc[0]
                
                # Base price for the day
                open_price = day_data['open']
                close_price = day_data['close']
                
                # Generate minute-by-minute prices
                minute_volatility = volatility / np.sqrt(390)  # 390 minutes in trading day
                
                # Create price path from open to close
                price_path = np.linspace(open_price, close_price, len(times))
                
                # Add random noise
                noise = np.random.normal(0, minute_volatility * open_price, size=len(times))
                minute_prices = price_path + np.cumsum(noise)
                
                # Ensure high and low are respected
                minute_high = np.maximum.accumulate(minute_prices)
                minute_low = np.minimum.accumulate(minute_prices)
                
                scale_high = day_data['high'] / minute_high[-1]
                scale_low = day_data['low'] / minute_low[-1]
                
                minute_high *= scale_high
                minute_low *= scale_low
                
                # Create minute-by-minute OHLCV data
                intraday_df = pd.DataFrame({
                    'datetime': times,
                    'open': minute_prices,
                    'high': minute_high,
                    'low': minute_low,
                    'close': minute_prices,
                    'volume': np.random.randint(1000, 100000, size=len(times))
                })
                
                # Set index to datetime
                intraday_df.set_index('datetime', inplace=True)
                
                # Calculate VWAP
                typical_price = (intraday_df['high'] + intraday_df['low'] + intraday_df['close']) / 3
                intraday_df['vwap'] = (typical_price * intraday_df['volume']).cumsum() / intraday_df['volume'].cumsum()
                
                # Store intraday data
                intraday_data[date.strftime('%Y-%m-%d')] = intraday_df
            
            # Store data for this symbol
            self.price_data[symbol] = {
                'daily': daily_data,
                'intraday': intraday_data
            }
        
        logger.info("Mock price data generation complete.")
    
    def generate_news_data(self):
        """Generate synthetic news data for backtesting."""
        logger.info("Generating mock news data...")
        
        # News templates for stock-specific news
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
        
        # News sources
        sources = ["Reuters", "Bloomberg", "CNBC", "Yahoo Finance", "MarketWatch"]
        
        # Generate news for each stock
        for symbol in self.stock_universe:
            self.news_data[symbol] = []
            
            # Generate all dates between start_date and end_date
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            
            # Skip weekends
            dates = [date for date in dates if date.weekday() < 5]
            
            for date in dates:
                # Random number of news items (0-3)
                num_items = random.randint(0, 3)
                
                for i in range(num_items):
                    # Pick a random template and source
                    template = random.choice(news_templates)
                    source = random.choice(sources)
                    
                    # Generate a headline
                    headline = template.format(symbol=symbol)
                    
                    # Generate a random time during the trading day
                    hours = random.randint(4, 20)  # 4 AM to 8 PM
                    minutes = random.randint(0, 59)
                    news_time = date.replace(hour=hours, minute=minutes)
                    
                    # Create a dummy URL
                    url = f"https://example.com/news/{symbol.lower()}/{date.strftime('%Y%m%d')}/{i}"
                    
                    # Create the news item
                    news_item = {
                        'headline': headline,
                        'source': source,
                        'url': url,
                        'date': news_time,
                        'score': random.randint(1, 10)
                    }
                    
                    # Add to news data
                    self.news_data[symbol].append(news_item)
        
        # Create general market news
        self.market_news = []
        
        # Market news templates
        market_templates = [
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
        
        # Generate market news for each trading day
        for date in [d for d in pd.date_range(start=self.start_date, end=self.end_date, freq='D') if d.weekday() < 5]:
            # Generate 1-3 market news items per day
            num_items = random.randint(1, 3)
            
            for i in range(num_items):
                template = random.choice(market_templates)
                source = random.choice(sources)
                
                # Generate a random time
                hours = random.randint(4, 20)
                minutes = random.randint(0, 59)
                news_time = date.replace(hour=hours, minute=minutes)
                
                # Create a dummy URL
                url = f"https://example.com/market-news/{date.strftime('%Y%m%d')}/{i}"
                
                # Create the news item
                news_item = {
                    'headline': template,
                    'source': source,
                    'url': url,
                    'date': news_time,
                    'score': random.randint(1, 10)
                }
                
                # Add to market news
                self.market_news.append(news_item)
        
        logger.info("Mock news data generation complete.")
    
    def get_price_data(self, symbol):
        """
        Get price data for a specific symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
            
        Returns:
        --------
        dict: Price data for the specified symbol
        """
        return self.price_data.get(symbol, None)
    
    def get_news_data(self, symbol):
        """
        Get news data for a specific symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
            
        Returns:
        --------
        list: News data for the specified symbol
        """
        return self.news_data.get(symbol, [])
    
    def get_market_news(self):
        """
        Get general market news.
        
        Returns:
        --------
        list: General market news
        """
        return self.market_news
