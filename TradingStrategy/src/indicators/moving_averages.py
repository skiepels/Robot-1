"""
Moving Averages Module

This module implements various moving average indicators for the trading strategy,
including the 9, 20, and 200 period EMAs.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MovingAverages:
    """
    Calculates and evaluates moving average indicators.
    
    This class provides methods to calculate different types of moving averages
    and evaluate price action relative to these averages.
    """
    
    def __init__(self):
        """Initialize the MovingAverages class."""
        pass
    
    def calculate_sma(self, data, period=20, column='close'):
        """
        Calculate Simple Moving Average (SMA).
        
        Parameters:
        -----------
        data: pandas.DataFrame or pandas.Series
            Price data
        period: int
            Moving average period
        column: str
            Column name to use if data is a DataFrame
            
        Returns:
        --------
        pandas.Series: Simple Moving Average
        """
        try:
            if isinstance(data, pd.DataFrame) and column in data.columns:
                prices = data[column]
            else:
                prices = data
                
            return prices.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series(index=data.index if hasattr(data, 'index') else None)
    
    def calculate_ema(self, data, period=20, column='close'):
        """
        Calculate Exponential Moving Average (EMA).
        
        Parameters:
        -----------
        data: pandas.DataFrame or pandas.Series
            Price data
        period: int
            Moving average period
        column: str
            Column name to use if data is a DataFrame
            
        Returns:
        --------
        pandas.Series: Exponential Moving Average
        """
        try:
            if isinstance(data, pd.DataFrame) and column in data.columns:
                prices = data[column]
            else:
                prices = data
                
            return prices.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series(index=data.index if hasattr(data, 'index') else None)
    
    def add_moving_averages(self, df):
        """
        Add key moving averages to a DataFrame.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with moving averages added
        """
        try:
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Add 9-period EMA
            result['ema9'] = self.calculate_ema(result, period=9)
            
            # Add 20-period EMA
            result['ema20'] = self.calculate_ema(result, period=20)
            
            # Add 200-period EMA
            result['ema200'] = self.calculate_ema(result, period=200)
            
            return result
        except Exception as e:
            logger.error(f"Error adding moving averages: {e}")
            return df
    
    def is_above_ema(self, price, ema_value):
        """
        Check if price is above the EMA.
        
        Parameters:
        -----------
        price: float
            Current price
        ema_value: float
            EMA value
            
        Returns:
        --------
        bool: True if price is above EMA
        """
        if price is None or ema_value is None or np.isnan(ema_value):
            return False
            
        return price > ema_value
    
    def is_bullish_ema_alignment(self, df):
        """
        Check if EMAs are in bullish alignment (9 > 20 > 200).
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with EMA columns
            
        Returns:
        --------
        bool: True if EMAs are in bullish alignment
        """
        if df is None or df.empty:
            return False
            
        # Make sure we have the necessary columns
        required_columns = ['ema9', 'ema20', 'ema200']
        if not all(col in df.columns for col in required_columns):
            # Calculate EMAs if they don't exist
            df_with_emas = self.add_moving_averages(df)
        else:
            df_with_emas = df
        
        # Get the most recent values
        latest = df_with_emas.iloc[-1]
        
        # Check if values are available and not NaN
        if any(np.isnan(latest[col]) for col in required_columns):
            return False
        
        # Check bullish alignment: 9 > 20 > 200
        return latest['ema9'] > latest['ema20'] > latest['ema200']
    
    def is_price_above_key_emas(self, df):
        """
        Check if price is above key EMAs (9 and 20).
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with EMA columns
            
        Returns:
        --------
        bool: True if price is above key EMAs
        """
        if df is None or df.empty:
            return False
            
        # Make sure we have the necessary columns
        required_columns = ['ema9', 'ema20', 'close']
        if not all(col in df.columns for col in required_columns):
            # Calculate EMAs if they don't exist
            df_with_emas = self.add_moving_averages(df)
        else:
            df_with_emas = df
        
        # Get the most recent values
        latest = df_with_emas.iloc[-1]
        
        # Check if values are available and not NaN
        if any(np.isnan(latest[col]) for col in required_columns):
            return False
        
        # Check if price is above both 9 and 20 EMAs
        return latest['close'] > latest['ema9'] and latest['close'] > latest['ema20']