"""
Market Data Provider

This module handles fetching and processing market data for the day trading strategy.
It implements methods to get stock prices, volume, and other market data needed
for Ross Cameron's momentum trading approach.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
import time
from src.data.stock import Stock

logger = logging.getLogger(__name__)


class MarketDataProvider:
    def __init__(self, api_key=None):
        """
        Initialize the market data provider.
        
        Parameters:
        -----------
        api_key: str, optional
            API key for premium data services (if used)
        """
        self.api_key = api_key
        self.cached_data = {}  # Cache for frequently accessed data
        self.cache_expiry = {}  # Expiry timestamps for cached data
        
    def get_tradable_stocks(self):
        """
        Get a list of all tradable stocks in the market.
        
        Returns:
        --------
        list: Stock objects for all available stocks
        """
        logger.info("Fetching tradable stocks...")
        
        # Check cache first
        cache_key = 'tradable_stocks'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached tradable stocks list")
            return self.cached_data[cache_key]
        
        try:
            # Attempt to get most active stocks from Yahoo Finance
            most_active_df = self._fetch_most_active_stocks()
            
            # Convert to Stock objects
            stocks = []
            for _, row in most_active_df.iterrows():
                stock = Stock(row['Symbol'], row.get('Name', None))
                stock.current_price = row.get('Price', 0.0)
                stock.current_volume = row.get('Volume', 0)
                stock.change_today_percent = row.get('Change', 0.0)
                
                # Get additional data for each stock
                self._enrich_stock_data(stock)
                
                stocks.append(stock)
            
            # Cache the results
            self._cache_data(cache_key, stocks, expiry_seconds=3600)  # 1 hour expiry
            
            logger.info(f"Fetched {len(stocks)} tradable stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching tradable stocks: {e}")
            
            # Fallback to a small sample of known active stocks
            fallback_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD']
            
            stocks = []
            for symbol in fallback_symbols:
                stock = Stock(symbol)
                self._enrich_stock_data(stock)
                stocks.append(stock)
                
            return stocks
    
    def get_gappers(self, min_gap_pct=5.0, min_price=1.0, max_price=20.0):
        """
        Get stocks that are gapping up significantly in pre-market.
        
        Parameters:
        -----------
        min_gap_pct: float
            Minimum gap percentage to include
        min_price: float
            Minimum stock price to consider
        max_price: float
            Maximum stock price to consider
            
        Returns:
        --------
        list: Stock objects that are gapping up
        """
        logger.info(f"Fetching stocks gapping up at least {min_gap_pct}%...")
        
        # Check cache first
        cache_key = f'gappers_{min_gap_pct}_{min_price}_{max_price}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached gappers list")
            return self.cached_data[cache_key]
        
        try:
            # First, get all tradable stocks
            all_stocks = self.get_tradable_stocks()
            
            # Filter for stocks in the price range
            price_filtered = [
                stock for stock in all_stocks 
                if min_price <= stock.current_price <= max_price
            ]
            
            # For each stock, check pre-market data to calculate gap
            gappers = []
            for stock in price_filtered:
                # Get pre-market data
                self._update_premarket_data(stock)
                
                # Check if it's gapping up enough
                if stock.gap_percent >= min_gap_pct:
                    gappers.append(stock)
            
            # Sort by gap percentage (descending)
            gappers.sort(key=lambda x: x.gap_percent, reverse=True)
            
            # Cache the results
            self._cache_data(cache_key, gappers, expiry_seconds=300)  # 5 min expiry
            
            logger.info(f"Found {len(gappers)} stocks gapping up at least {min_gap_pct}%")
            return gappers
            
        except Exception as e:
            logger.error(f"Error fetching gappers: {e}")
            return []
    
    def get_high_relative_volume_stocks(self, min_rel_volume=5.0):
        """
        Get stocks trading with high relative volume compared to their 50-day average.
        
        Parameters:
        -----------
        min_rel_volume: float
            Minimum relative volume ratio to include
            
        Returns:
        --------
        list: Stock objects with high relative volume
        """
        logger.info(f"Fetching stocks with at least {min_rel_volume}x relative volume...")
        
        # Check cache first
        cache_key = f'high_rel_vol_{min_rel_volume}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached high relative volume stocks list")
            return self.cached_data[cache_key]
        
        try:
            # Get all tradable stocks
            all_stocks = self.get_tradable_stocks()
            
            # Update volume data for each stock
            for stock in all_stocks:
                self._update_volume_data(stock)
            
            # Filter for stocks with high relative volume
            high_rel_vol_stocks = [
                stock for stock in all_stocks 
                if stock.relative_volume >= min_rel_volume
            ]
            
            # Sort by relative volume (descending)
            high_rel_vol_stocks.sort(key=lambda x: x.relative_volume, reverse=True)
            
            # Cache the results
            self._cache_data(cache_key, high_rel_vol_stocks, expiry_seconds=300)  # 5 min expiry
            
            logger.info(f"Found {len(high_rel_vol_stocks)} stocks with at least {min_rel_volume}x relative volume")
            return high_rel_vol_stocks
            
        except Exception as e:
            logger.error(f"Error fetching high relative volume stocks: {e}")
            return []
    
    def get_intraday_data(self, symbol, interval='1m', lookback_days=1):
        """
        Get intraday price data for a stock.
        
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
        logger.info(f"Fetching {interval} data for {symbol} for the past {lookback_days} days...")
        
        # Check cache first
        cache_key = f'intraday_{symbol}_{interval}_{lookback_days}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached intraday data")
            return self.cached_data[cache_key]
        
        try:
            # Make sure interval is a string
            if not isinstance(interval, str):
                interval = '1m'

            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            # Fetch data using yfinance
            data = yf.download(
                symbol,
                start=start_time,
                end=end_time,
                interval=interval,
                prepost=True,  # Include pre/post market data
                progress=False
            )
            
            # Handle empty data
            if data.empty:
                logger.warning(f"No intraday data available for {symbol}")
                # Return an empty DataFrame with the right columns
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Normalize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add additional columns if needed
            if 'vwap' not in data.columns:
                # Calculate VWAP
                data['vwap'] = self._calculate_vwap(data)
            
            # Cache the results (short expiry for intraday data)
            self._cache_data(cache_key, data, expiry_seconds=60)  # 1 min expiry
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
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
        try:
            # Try to get recent data
            recent_data = self.get_intraday_data(symbol, interval='1m', lookback_days=1)
            
            if not recent_data.empty:
                return recent_data['close'].iloc[-1]
            
            # If intraday data not available, try ticker info
            ticker = yf.Ticker(symbol)
            return ticker.info.get('regularMarketPrice', None)
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_low_float_stocks(self, max_float=10_000_000):
        """
        Get stocks with a low float (number of shares available for trading).
        
        Parameters:
        -----------
        max_float: int
            Maximum number of shares in the float
            
        Returns:
        --------
        list: Stock objects with float below the specified threshold
        """
        logger.info(f"Fetching stocks with float below {max_float:,} shares...")
        
        # Check cache first
        cache_key = f'low_float_{max_float}'
        if self._is_cache_valid(cache_key):
            logger.info("Using cached low float stocks list")
            return self.cached_data[cache_key]
        
        try:
            # Get all tradable stocks
            all_stocks = self.get_tradable_stocks()
            
            # Update float data for each stock
            for stock in all_stocks:
                self._update_float_data(stock)
            
            # Filter for stocks with low float
            low_float_stocks = [
                stock for stock in all_stocks 
                if 0 < stock.shares_float <= max_float  # Exclude stocks with unknown float (0)
            ]
            
            # Sort by float size (ascending)
            low_float_stocks.sort(key=lambda x: x.shares_float)
            
            # Cache the results
            self._cache_data(cache_key, low_float_stocks, expiry_seconds=86400)  # 24 hour expiry
            
            logger.info(f"Found {len(low_float_stocks)} stocks with float below {max_float:,} shares")
            return low_float_stocks
            
        except Exception as e:
            logger.error(f"Error fetching low float stocks: {e}")
            return []
    
    def _fetch_most_active_stocks(self):
        """Fetch most active stocks for the day"""
        try:
            # Use yfinance to get most active stocks
            # This is a simulated version for demonstration
            
            # In a real implementation, you might use a different data source
            tickers = yf.Tickers('AAPL MSFT AMZN TSLA GOOGL META AMD NVDA')
            
            # Create a DataFrame with the most active stocks
            data = []
            for ticker, ticker_info in tickers.tickers.items():
                # Get basic info
                info = ticker_info.info
                
                data.append({
                    'Symbol': ticker,
                    'Name': info.get('shortName', ticker),
                    'Price': info.get('regularMarketPrice', 0.0),
                    'Volume': info.get('regularMarketVolume', 0),
                    'Change': info.get('regularMarketChangePercent', 0.0)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching most active stocks: {e}")
            
            # Return an empty DataFrame
            return pd.DataFrame(columns=['Symbol', 'Name', 'Price', 'Volume', 'Change'])
    
    def _enrich_stock_data(self, stock):
        """Add additional data to a stock object"""
        try:
            # Fetch basic data using yfinance
            ticker = yf.Ticker(stock.symbol)
            info = ticker.info
            
            # Update price data
            stock.current_price = info.get('regularMarketPrice', 0.0)
            stock.open_price = info.get('regularMarketOpen', 0.0)
            stock.high_price = info.get('regularMarketDayHigh', 0.0)
            stock.low_price = info.get('regularMarketDayLow', 0.0)
            stock.previous_close = info.get('previousClose', 0.0)
            
            # Update volume data
            stock.current_volume = info.get('regularMarketVolume', 0)
            stock.avg_volume_50d = info.get('averageDailyVolume10Day', 0)  # Using 10-day as an approximation
            
            if stock.avg_volume_50d > 0:
                stock.relative_volume = stock.current_volume / stock.avg_volume_50d
            
            # Update share data
            stock.shares_outstanding = info.get('sharesOutstanding', 0)
            stock.shares_float = info.get('floatShares', 0)
            stock.shares_short = info.get('sharesShort', 0)
            
            # Update price change data
            if stock.previous_close > 0:
                stock.change_today = stock.current_price - stock.previous_close
                stock.change_today_percent = (stock.change_today / stock.previous_close) * 100
            
            # If open price is available, calculate gap percentage
            if stock.open_price > 0 and stock.previous_close > 0:
                stock.gap_percent = (stock.open_price - stock.previous_close) / stock.previous_close * 100
                
        except Exception as e:
            logger.warning(f"Error enriching data for {stock.symbol}: {e}")
    
    def _update_premarket_data(self, stock):
        """Update pre-market data for a stock"""
        try:
            # For simulation purposes, we'll just set a random gap percentage
            # In a real implementation, you would fetch actual pre-market data
            
            if stock.previous_close > 0:
                # Simulate a gap between 0% and 20%
                import random
                gap_percent = random.uniform(0, 20)
                stock.gap_percent = gap_percent
                
                # Calculate pre-market price based on gap
                premarket_price = stock.previous_close * (1 + gap_percent / 100)
                stock.premarket_open = premarket_price
                stock.premarket_close = premarket_price
                
            logger.debug(f"Simulated pre-market data for {stock.symbol}: gap {stock.gap_percent:.2f}%")
                
        except Exception as e:
            logger.warning(f"Error updating pre-market data for {stock.symbol}: {e}")
    
    def _update_volume_data(self, stock):
        """Update volume data for a stock with real calculations"""
        try:
            # Get historical data for volume calculation
            ticker = yf.Ticker(stock.symbol)
            
            # Get 50 days of historical data
            hist_data = ticker.history(period="50d")
            
            if not hist_data.empty and 'Volume' in hist_data.columns:
                # Calculate 50-day average volume
                stock.avg_volume_50d = hist_data['Volume'].mean()
                
                # Get current volume (most recent trading day)
                stock.current_volume = hist_data['Volume'].iloc[-1]
                
                # Calculate relative volume
                if stock.avg_volume_50d > 0:
                    stock.relative_volume = stock.current_volume / stock.avg_volume_50d
                else:
                    stock.relative_volume = 0.0
                    
                logger.debug(f"Updated volume data for {stock.symbol}: " +
                        f"current volume: {stock.current_volume:,}, " +
                        f"50-day avg: {stock.avg_volume_50d:,.0f}, " +
                        f"relative volume: {stock.relative_volume:.2f}x")
            else:
                logger.warning(f"No volume data available for {stock.symbol}")
                stock.relative_volume = 0.0
                
        except Exception as e:
            logger.warning(f"Error updating volume data for {stock.symbol}: {e}")
            stock.relative_volume = 0.0
    
    def _update_float_data(self, stock):
        """Update float data for a stock"""
        try:
            # For simulation purposes, we'll set a random float size
            # In a real implementation, you would fetch this from a data provider
            
            # Simulate a float between 1M and 50M shares
            import random
            shares_float = random.randint(1_000_000, 50_000_000)
            stock.shares_float = shares_float
            
            # Simulate shares outstanding (usually higher than float)
            stock.shares_outstanding = shares_float * random.uniform(1.1, 2.0)
            
            # Simulate short interest
            stock.shares_short = shares_float * random.uniform(0, 0.2)
            
            # Calculate short ratio
            if stock.shares_float > 0:
                stock.short_ratio = stock.shares_short / stock.shares_float
                
            logger.debug(f"Simulated float data for {stock.symbol}: {stock.shares_float:,} shares")
                
        except Exception as e:
            logger.warning(f"Error updating float data for {stock.symbol}: {e}")
    
    def _calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price (VWAP)"""
        try:
            if 'volume' not in data.columns:
                return pd.Series(index=data.index)
                
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            return vwap
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return pd.Series(index=data.index)
    
    def _cache_data(self, key, data, expiry_seconds=300):
        """Cache data with an expiry time"""
        self.cached_data[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=expiry_seconds)
    
    def _is_cache_valid(self, key):
        """Check if cached data exists and is still valid"""
        if key not in self.cached_data or key not in self.cache_expiry:
            return False
            
        return datetime.now() < self.cache_expiry[key]