"""
MACD (Moving Average Convergence Divergence) Module

This module implements the MACD indicator for the trading strategy.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MACD:
    """
    Calculates and evaluates the MACD indicator.
    
    The Moving Average Convergence Divergence (MACD) is a trend-following
    momentum indicator that shows the relationship between two moving averages
    of a security's price.
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Initialize the MACD indicator.
        
        Parameters:
        -----------
        fast_period: int
            Period for the fast EMA
        slow_period: int
            Period for the slow EMA
        signal_period: int
            Period for the signal line
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data, column='close'):
        """
        Calculate MACD components: MACD line, signal line, and histogram.
        
        Parameters:
        -----------
        data: pandas.DataFrame or pandas.Series
            Price data
        column: str
            Column name to use if data is a DataFrame
            
        Returns:
        --------
        tuple: (MACD line, signal line, histogram)
        """
        try:
            if isinstance(data, pd.DataFrame) and column in data.columns:
                prices = data[column]
            else:
                prices = data
            
            # Calculate fast and slow EMAs
            fast_ema = prices.ewm(span=self.fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            empty_series = pd.Series(index=data.index if hasattr(data, 'index') else None)
            return empty_series, empty_series, empty_series
    
    def add_macd(self, df, column='close'):
        """
        Add MACD components to a DataFrame.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data
        column: str
            Column name to use for calculations
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with MACD components added
        """
        try:
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Calculate MACD components
            macd_line, signal_line, histogram = self.calculate(result, column)
            
            # Add to DataFrame
            result['macd_line'] = macd_line
            result['macd_signal'] = signal_line
            result['macd_histogram'] = histogram
            
            return result
        
        except Exception as e:
            logger.error(f"Error adding MACD to DataFrame: {e}")
            return df
    
    def is_bullish_crossover(self, df, n_periods=1):
        """
        Check if a bullish MACD crossover occurred in the last n periods.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with MACD columns
        n_periods: int
            Number of periods to look back
            
        Returns:
        --------
        bool: True if a bullish crossover occurred
        """
        if df is None or df.empty or n_periods < 1:
            return False
        
        # Make sure we have the necessary columns
        required_columns = ['macd_line', 'macd_signal']
        if not all(col in df.columns for col in required_columns):
            # Calculate MACD if the columns don't exist
            df_with_macd = self.add_macd(df)
        else:
            df_with_macd = df
        
        # Get the relevant periods
        lookback = min(n_periods + 1, len(df_with_macd))
        recent_data = df_with_macd.iloc[-lookback:]
        
        # Check for crossover: MACD line crosses above signal line
        for i in range(1, len(recent_data)):
            prev = recent_data.iloc[i-1]
            curr = recent_data.iloc[i]
            
            # Check if MACD line crossed above signal line
            if (prev['macd_line'] <= prev['macd_signal'] and 
                curr['macd_line'] > curr['macd_signal']):
                return True
        
        return False
    
    def is_bullish_momentum(self, df):
        """
        Check if MACD shows bullish momentum.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with MACD columns
            
        Returns:
        --------
        bool: True if MACD shows bullish momentum
        """
        if df is None or df.empty:
            return False
        
        # Make sure we have the necessary columns
        required_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        if not all(col in df.columns for col in required_columns):
            # Calculate MACD if the columns don't exist
            df_with_macd = self.add_macd(df)
        else:
            df_with_macd = df
        
        # Get the most recent values
        if len(df_with_macd) < 2:
            return False
            
        latest = df_with_macd.iloc[-1]
        previous = df_with_macd.iloc[-2]
        
        # Check if values are available and not NaN
        if any(np.isnan(latest[col]) for col in required_columns):
            return False
        
        # Bullish criteria:
        # 1. MACD line is above signal line
        # 2. MACD histogram is positive and increasing
        
        macd_above_signal = latest['macd_line'] > latest['macd_signal']
        
        histogram_positive = latest['macd_histogram'] > 0
        histogram_increasing = latest['macd_histogram'] > previous['macd_histogram']
        
        return macd_above_signal and histogram_positive and histogram_increasing
    
    def is_macd_negative(self, df):
        """
        Check if MACD is negative.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            Price data with MACD columns
            
        Returns:
        --------
        bool: True if MACD line is negative
        """
        if df is None or df.empty:
            return False
        
        # Make sure we have the necessary columns
        if 'macd_line' not in df.columns:
            # Calculate MACD if the column doesn't exist
            df_with_macd = self.add_macd(df)
        else:
            df_with_macd = df
        
        # Get the most recent value
        latest = df_with_macd.iloc[-1]
        
        # Check if MACD line is negative
        return latest['macd_line'] < 0