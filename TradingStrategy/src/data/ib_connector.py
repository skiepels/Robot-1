"""
Interactive Brokers Connector

This module handles connecting to the Interactive Brokers API
and retrieving market data.
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Contract, Stock, Forex, Index, Bars, util

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class IBConnector:
    """
    A connector class for Interactive Brokers API.
    Handles connection, data retrieval, and other IB-related functionality.
    """
    
    def __init__(self, host=None, port=None, client_id=None):
        """
        Initialize the IB connector.
        
        Parameters:
        -----------
        host: str
            TWS/IB Gateway host (default: from .env)
        port: int
            TWS/IB Gateway port (default: from .env)
        client_id: int
            Client ID for API connection (default: from .env)
        """
        # Use provided values or get from environment variables
        self.host = host or os.getenv('IB_HOST', '127.0.0.1')
        self.port = int(port or os.getenv('IB_PORT', 7497))
        self.client_id = int(client_id or os.getenv('IB_CLIENT_ID', 1))
        
        # Initialize IB instance
        self.ib = IB()
        self.connected = False
        
        # Track request IDs to avoid duplicates
        self.last_req_id = 0
        
        # Cache for frequently accessed data
        self.data_cache = {}
        self.cache_expiry = {}
        
        logger.info(f"IBConnector initialized with host={self.host}, port={self.port}, client_id={self.client_id}")

    def connect(self, max_attempts=3, retry_delay=2):
        """
        Connect to Interactive Brokers TWS/Gateway.
        
        Parameters:
        -----------
        max_attempts: int
            Maximum number of connection attempts
        retry_delay: int
            Seconds to wait between attempts
            
        Returns:
        --------
        bool: True if connection successful, False otherwise
        """
        if self.connected and self.ib.isConnected():
            logger.info("Already connected to IB")
            return True
            
        logger.info(f"Connecting to IB at {self.host}:{self.port}")
        
        # Try to connect
        for attempt in range(max_attempts):
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                if self.ib.isConnected():
                    self.connected = True
                    logger.info("Successfully connected to Interactive Brokers")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        logger.error(f"Failed to connect to IB after {max_attempts} attempts")
        return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.connected:
            logger.info("Disconnecting from IB")
            self.ib.disconnect()
            self.connected = False
    
    def create_stock_contract(self, symbol, exchange='SMART', currency='USD'):
        """
        Create a stock contract object.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        exchange: str
            Exchange (default: 'SMART' for smart routing)
        currency: str
            Currency (default: 'USD')
            
        Returns:
        --------
        Contract: IB contract object
        """
        return Stock(symbol, exchange, currency)
    
    def get_historical_data(self, symbol, duration='1 D', bar_size='1 min', 
                           what_to_show='TRADES', use_rth=True):
        """
        Get historical price data for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        duration: str
            Time duration for data (e.g., '1 D', '1 W', '1 M')
        bar_size: str
            Bar size (e.g., '1 min', '5 mins', '1 hour')
        what_to_show: str
            Data type (e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK')
        use_rth: bool
            Use regular trading hours only
            
        Returns:
        --------
        pandas.DataFrame: Historical OHLCV data
        """
        if not self.ensure_connection():
            return pd.DataFrame()
            
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        try:
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract, 
                endDateTime='', 
                durationStr=duration,
                barSizeSetting=bar_size, 
                whatToShow=what_to_show,
                useRTH=use_rth
            )
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Rename columns to match our convention
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol, timeout=5):
        """
        Get current market price for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        timeout: int
            Timeout in seconds
            
        Returns:
        --------
        float: Current market price or None if not available
        """
        if not self.ensure_connection():
            return None
            
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        try:
            # Request market data
            self.ib.reqMktData(contract)
            
            # Wait for data to arrive
            end_time = time.time() + timeout
            while time.time() < end_time:
                self.ib.sleep(0.1)
                ticker = self.ib.ticker(contract)
                
                # Check for valid price data
                if ticker.marketPrice() > 0:
                    price = ticker.marketPrice()
                    logger.info(f"Current price for {symbol}: ${price:.2f}")
                    
                    # Cancel market data to avoid unnecessary data feed
                    self.ib.cancelMktData(contract)
                    return price
            
            # Cancel market data if timeout
            self.ib.cancelMktData(contract)
            logger.warning(f"Timeout getting current price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_current_volume(self, symbol, timeout=5):
        """
        Get current volume for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        timeout: int
            Timeout in seconds
            
        Returns:
        --------
        int: Current volume or None if not available
        """
        if not self.ensure_connection():
            return None
            
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        try:
            # Request market data
            self.ib.reqMktData(contract)
            
            # Wait for data to arrive
            end_time = time.time() + timeout
            while time.time() < end_time:
                self.ib.sleep(0.1)
                ticker = self.ib.ticker(contract)
                
                # Check for valid volume data
                if ticker.volume > 0:
                    volume = ticker.volume
                    logger.info(f"Current volume for {symbol}: {volume}")
                    
                    # Cancel market data to avoid unnecessary data feed
                    self.ib.cancelMktData(contract)
                    return volume
            
            # Cancel market data if timeout
            self.ib.cancelMktData(contract)
            logger.warning(f"Timeout getting current volume for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current volume for {symbol}: {str(e)}")
            return None
    
    def get_contract_details(self, symbol):
        """
        Get contract details for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        dict: Contract details
        """
        if not self.ensure_connection():
            return {}
            
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        try:
            # Request contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                # Extract relevant information
                detail = details[0]
                info = {
                    'symbol': symbol,
                    'long_name': detail.longName,
                    'industry': detail.industry,
                    'category': detail.category,
                    'subcategory': detail.subcategory,
                    'price_magnifier': detail.priceMagnifier,
                    'min_tick': detail.minTick,
                    'order_types': detail.orderTypes.split(','),
                    'valid_exchanges': detail.validExchanges.split(','),
                    'market_name': detail.marketName,
                    'contract': detail.contract
                }
                
                # Extract float information if available
                if hasattr(detail.marketRuleIds, 'float'):
                    info['float'] = detail.marketRuleIds.float
                
                logger.info(f"Retrieved contract details for {symbol}")
                return info
            else:
                logger.warning(f"No contract details found for {symbol}")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting contract details for {symbol}: {str(e)}")
            return {}
    
    def ensure_connection(self):
        """Ensure connection to IB is active, reconnect if necessary."""
        if not self.connected or not self.ib.isConnected():
            logger.info("Connection to IB not active, attempting to reconnect")
            return self.connect()
        return True
    
    def run_event_loop(self):
        """Run the IB event loop in the current thread."""
        if self.ensure_connection():
            logger.info("Running IB event loop")
            self.ib.run()
    
    def __del__(self):
        """Ensure disconnection when object is destroyed."""
        self.disconnect()