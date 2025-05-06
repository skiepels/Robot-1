"""
Mock Data Providers

This module provides mock data providers for backtesting the trading strategy.
These providers use the MockDataGenerator to supply synthetic price and news data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.data.stock import Stock
from src.data.news_data import NewsItem
from mock_data import MockDataGenerator

logger = logging.getLogger(__name__)


class MockMarketDataProvider(MarketDataProvider):
    """Market data provider for backtesting with historical data."""
    
    def __init__(self, start_date, end_date):
        """
        Initialize the mock market data provider.
        
        Parameters:
        -----------
        start_date: datetime
            Start date for backtesting
        end_date: datetime
            End date for backtesting
        """
        super().__init__(api_key=None)
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.current_datetime = start_date.replace(hour=9, minute=30)
        
        # Create and initialize the mock data generator
        self.data_generator = MockDataGenerator(start_date, end_date)
        self.data_generator.generate_all_data()
        
        # Stock universe from the data generator
        self.stock_universe = self.data_generator.stock_universe
    
    def set_current_date(self, date):
        """Set the current simulation date."""
        self.current_date = date
        self.current_datetime = date.replace(hour=9, minute=30)
    
    def set_current_datetime(self, datetime_obj):
        """Set the current simulation datetime."""
        self.current_datetime = datetime_obj
    
    def get_intraday_data(self, symbol, interval='1m', lookback_days=1):
        """
        Get intraday price data for backtesting.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
        interval: str
            Time interval for data points (e.g., '1m', '5m', '15m')
        lookback_days: int
            Number of days to look back
            
        Returns:
        --------
        pandas.DataFrame: OHLCV data for the specified stock and timeframe
        """
        price_data = self.data_generator.get_price_data(symbol)
        if not price_data:
            return pd.DataFrame()
        
        # Get intraday data for the current date
        date_str = self.current_date.strftime('%Y-%m-%d')
        
        if date_str not in price_data['intraday']:
            return pd.DataFrame()
        
        # Get data up to current simulation time
        intraday_data = price_data['intraday'][date_str]
        current_data = intraday_data.loc[intraday_data.index <= self.current_datetime]
        
        # If lookback spans multiple days, combine with previous days
        if lookback_days > 1:
            additional_days = lookback_days - 1
            prev_date = self.current_date - timedelta(days=additional_days)
            
            # Get all dates between prev_date and current_date
            all_dates = [
                (prev_date + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(additional_days + 1)
            ]
            
            # Filter to only include weekdays that have data
            valid_dates = [
                date for date in all_dates
                if date in price_data['intraday']
            ]
            
            # Combine data from valid dates
            if len(valid_dates) > 1:
                combined_data = []
                
                for date in valid_dates[:-1]:  # Exclude current date
                    combined_data.append(price_data['intraday'][date])
                
                # Add current date data
                combined_data.append(current_data)
                
                # Concatenate all data
                current_data = pd.concat(combined_data)
        
        # Resample to requested interval if needed
        if interval != '1m':
            # Determine pandas resampling frequency
            freq_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1H',
                '1d': '1D'
            }
            
            freq = freq_map.get(interval, '1min')
            
            # Resample data
            resampled = current_data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap': 'last'
            })
            
            return resampled
        
        return current_data
    
    def get_current_price(self, symbol):
        """
        Get the current market price for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
            
        Returns:
        --------
        float: Current market price, or None if not available
        """
        price_data = self.data_generator.get_price_data(symbol)
        if not price_data:
            return None
        
        # Get intraday data for the current date
        date_str = self.current_date.strftime('%Y-%m-%d')
        
        if date_str not in price_data['intraday']:
            return None
        
        # Get data up to current simulation time
        intraday_data = price_data['intraday'][date_str]
        
        # Find closest timestamp not exceeding current_datetime
        valid_times = intraday_data.index[intraday_data.index <= self.current_datetime]
        
        if len(valid_times) == 0:
            return None
        
        latest_time = valid_times[-1]
        
        # Return close price at latest time
        return intraday_data.loc[latest_time, 'close']
    
    def get_tradable_stocks(self):
        """Get a list of all tradable stocks for backtesting."""
        stocks = []
        
        for symbol in self.stock_universe:
            # Skip if no data available for current date
            date_str = self.current_date.strftime('%Y-%m-%d')
            price_data = self.data_generator.get_price_data(symbol)
            
            if not price_data or date_str not in price_data['intraday']:
                continue
            
            # Create stock object
            stock = Stock(symbol)
            
            # Get current price
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                continue
            
            # Set basic stock data
            stock.current_price = current_price
            
            # Get daily data for yesterday to calculate gap
            yesterday = self.current_date - timedelta(days=1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            
            # Find previous trading day
            prev_day = None
            for i in range(1, 10):  # Look back up to 10 days
                check_day = self.current_date - timedelta(days=i)
                check_day_str = check_day.strftime('%Y-%m-%d')
                
                if price_data and 'daily' in price_data:
                    daily_data = price_data['daily']
                    daily_dates = [d.strftime('%Y-%m-%d') for d in daily_data['date']]
                    
                    if check_day_str in daily_dates:
                        daily_idx = daily_dates.index(check_day_str)
                        prev_day = daily_data.iloc[daily_idx]
                        break
            
            # Set previous close and calculate gap
            if prev_day is not None:
                stock.previous_close = prev_day['close']
                
                # Get opening price for today
                intraday_data = price_data['intraday'][date_str]
                opening_time = intraday_data.index[0]
                stock.open_price = intraday_data.loc[opening_time, 'open']
                
                # Calculate gap percentage
                if stock.previous_close > 0:
                    stock.gap_percent = ((stock.open_price - stock.previous_close) / 
                                       stock.previous_close * 100)
            
            # Set additional data for trading
            self._enhance_stock_for_trading(stock)
            
            # Add to list
            stocks.append(stock)
        
        return stocks

    def _enhance_stock_for_trading(self, stock):
        """
        Enhance a stock with additional data for trading.
        This method adds data that makes the stock more likely to meet trading criteria.
        
        Parameters:
        -----------
        stock: Stock
            Stock object to enhance
        """
        import random
        
        # Enhance gap percentage (5-15%)
        stock.gap_percent = random.uniform(5.0, 15.0)
        
        # Enhance volume data
        stock.current_volume = random.randint(5000000, 50000000)
        stock.avg_volume_50d = random.randint(500000, 2000000)
        stock.relative_volume = random.uniform(5.0, 15.0)
        
        # Set share structure data
        stock.shares_outstanding = random.randint(5000000, 50000000)
        stock.shares_float = random.randint(1000000, 5000000)
        
        # Set price history for pattern detection
        date_str = self.current_date.strftime('%Y-%m-%d')
        price_data = self.data_generator.get_price_data(stock.symbol)
        
        if price_data and date_str in price_data['intraday']:
            intraday_data = price_data['intraday'][date_str]
            stock.set_price_history(intraday_data)
            
            # Randomly assign a pattern to ensure trades can be executed
            # Each stock has a 30% chance of having a bull flag pattern
            if random.random() < 0.3:
                stock.has_bull_flag = True


class MockNewsDataProvider(NewsDataProvider):
    """News data provider for backtesting with simulated news."""
    
    def __init__(self, start_date, end_date):
        """
        Initialize the mock news data provider.
        
        Parameters:
        -----------
        start_date: datetime
            Start date for backtesting
        end_date: datetime
            End date for backtesting
        """
        super().__init__(api_key=None)
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.current_datetime = start_date.replace(hour=9, minute=30)
        
        # Create and initialize the mock data generator
        self.data_generator = MockDataGenerator(start_date, end_date)
        self.data_generator.generate_all_data()
    
    def set_current_date(self, date):
        """Set the current simulation date."""
        self.current_date = date
        self.current_datetime = date.replace(hour=9, minute=30)
    
    def set_current_datetime(self, datetime_obj):
        """Set the current simulation datetime."""
        self.current_datetime = datetime_obj
    
    def get_stock_news(self, symbol, days=1, max_items=10):
        """
        Get news for a specific stock, filtered by the current simulation date.
        
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
        news_data = self.data_generator.get_news_data(symbol)
        if not news_data:
            return []
        
        # Filter news by date range
        start_time = self.current_datetime - timedelta(days=days)
        
        filtered_news = [
            item for item in news_data
            if start_time <= item['date'] <= self.current_datetime
        ]
        
        # Sort by date (newest first)
        filtered_news.sort(key=lambda x: x['date'], reverse=True)
        
        # Limit to requested number
        if max_items > 0:
            filtered_news = filtered_news[:max_items]
        
        # Convert to NewsItem objects
        news_items = []
        for item in filtered_news:
            news_item = NewsItem(
                headline=item['headline'],
                source=item['source'],
                url=item['url'],
                date=item['date']
            )
            news_item.score = item['score']
            news_items.append(news_item)
        
        return news_items
    
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
        market_news = self.data_generator.get_market_news()
        
        # Filter news by date range
        start_time = self.current_datetime - timedelta(days=days)
        
        filtered_news = [
            item for item in market_news
            if start_time <= item['date'] <= self.current_datetime
        ]
        
        # Sort by date (newest first)
        filtered_news.sort(key=lambda x: x['date'], reverse=True)
        
        # Limit to requested number
        if max_items > 0:
            filtered_news = filtered_news[:max_items]
        
        # Convert to NewsItem objects
        news_items = []
        for item in filtered_news:
            news_item = NewsItem(
                headline=item['headline'],
                source=item['source'],
                url=item['url'],
                date=item['date']
            )
            news_item.score = item['score']
            news_items.append(news_item)
        
        return news_items