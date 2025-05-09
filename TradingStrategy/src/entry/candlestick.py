"""
Candlestick Pattern Recognition Module

This module implements the detection of triple candlestick patterns:

Bullish Patterns:
- Morning Star
- Morning Doji Star
- Three White Soldiers
- Rising Three

Bearish Patterns:
- Evening Star
- Evening Doji Star
- Three Black Crows
- Falling Three
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """
    Detects various triple candlestick patterns for trade entry signals.
    
    This class contains methods to identify high-probability entry patterns
    using triple candlestick formations in price data.
    """
    
    def __init__(self):
        """Initialize the candlestick pattern detector."""
        pass
    
    def is_bullish_candle(self, candle):
        """
        Determine if a candle is bullish (close > open).
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'open' and 'close' values
            
        Returns:
        --------
        bool: True if bullish, False otherwise
        """
        return candle['close'] > candle['open']
    
    def is_bearish_candle(self, candle):
        """
        Determine if a candle is bearish (close < open).
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'open' and 'close' values
            
        Returns:
        --------
        bool: True if bearish, False otherwise
        """
        return candle['close'] < candle['open']
    
    def is_doji(self, data, index=-1, threshold=0.1):
        """
        Check if the candle at the given index is a doji.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data
        index: int
            Index of the candle to check (-1 means the last candle)
        threshold: float
            Maximum ratio of body to range to be considered a doji
            
        Returns:
        --------
        bool: True if the candle is a doji
        """
        candle = data.iloc[index]
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return False
        
        body_to_range_ratio = body_size / candle_range
        return body_to_range_ratio <= threshold
    
    def calculate_body_size(self, candle):
        """
        Calculate the absolute size of the candle body.
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'open' and 'close' values
            
        Returns:
        --------
        float: Absolute size of the candle body
        """
        return abs(candle['close'] - candle['open'])
    
    def calculate_range(self, candle):
        """
        Calculate the full range of a candle (high to low).
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'high' and 'low' values
            
        Returns:
        --------
        float: Range of the candle
        """
        return candle['high'] - candle['low']
    
    def calculate_upper_wick(self, candle):
        """
        Calculate the size of the upper wick/shadow.
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'open', 'close', and 'high' values
            
        Returns:
        --------
        float: Size of the upper wick
        """
        return candle['high'] - max(candle['open'], candle['close'])
    
    def calculate_lower_wick(self, candle):
        """
        Calculate the size of the lower wick/shadow.
        
        Parameters:
        -----------
        candle: pandas.Series
            Candle data with 'open', 'close', and 'low' values
            
        Returns:
        --------
        float: Size of the lower wick
        """
        return min(candle['open'], candle['close']) - candle['low']
    
    def detect_morning_star(self, data, doji_required=False):
        """
        Detect a Morning Star pattern (bullish reversal).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 3 candles
        doji_required: bool
            If True, requires the middle candle to be a doji (Morning Doji Star)
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 3:
            return False
        
        # Get the relevant candles
        first = data.iloc[-3]
        middle = data.iloc[-2]
        last = data.iloc[-1]
        
        # First candle should be bearish (close < open)
        if first['close'] >= first['open']:
            return False
        
        # Last candle should be bullish (close > open)
        if last['close'] <= last['open']:
            return False
        
        # Middle candle should have a small body
        first_body = abs(first['close'] - first['open'])
        middle_body = abs(middle['close'] - middle['open'])
        
        if middle_body > 0.3 * first_body:  # Middle body should be less than 30% of first body
            return False
        
        # If doji is required, check that middle candle is a doji
        if doji_required and not self.is_doji(data, -2):
            return False
        
        # Check if last candle closes well into the first candle's body
        first_midpoint = (first['open'] + first['close']) / 2
        if last['close'] <= first_midpoint:
            return False
        
        return True
    
    def detect_evening_star(self, data, doji_required=False):
        """
        Detect an Evening Star pattern (bearish reversal).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 3 candles
        doji_required: bool
            If True, requires the middle candle to be a doji (Evening Doji Star)
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 3:
            return False
        
        # Get the relevant candles
        first = data.iloc[-3]
        middle = data.iloc[-2]
        last = data.iloc[-1]
        
        # First candle should be bullish (close > open)
        if first['close'] <= first['open']:
            return False
        
        # Last candle should be bearish (close < open)
        if last['close'] >= last['open']:
            return False
        
        # Middle candle should have a small body
        first_body = abs(first['close'] - first['open'])
        middle_body = abs(middle['close'] - middle['open'])
        
        if middle_body > 0.3 * first_body:  # Middle body should be less than 30% of first body
            return False
        
        # If doji is required, check that middle candle is a doji
        if doji_required and not self.is_doji(data, -2):
            return False
        
        # Check if last candle closes well into the first candle's body
        first_midpoint = (first['open'] + first['close']) / 2
        if last['close'] >= first_midpoint:
            return False
        
        return True
    
    def detect_three_white_soldiers(self, data):
        """
        Detect a Three White Soldiers pattern (bullish continuation).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 3 candles
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 3:
            return False
        
        # Get the three candles
        candles = [data.iloc[-3], data.iloc[-2], data.iloc[-1]]
        
        # All three candles should be bullish
        if not all(candle['close'] > candle['open'] for candle in candles):
            return False
        
        # Each candle should close higher than the previous
        if not (candles[1]['close'] > candles[0]['close'] and candles[2]['close'] > candles[1]['close']):
            return False
        
        # Each candle should open within the previous candle's body
        if not (candles[1]['open'] > candles[0]['open'] and 
                candles[1]['open'] < candles[0]['close'] and
                candles[2]['open'] > candles[1]['open'] and
                candles[2]['open'] < candles[1]['close']):
            return False
        
        # Check for small upper shadows (less than 15% of body size)
        for candle in candles:
            body_size = candle['close'] - candle['open']
            upper_shadow = candle['high'] - candle['close']
            
            if upper_shadow > 0.15 * body_size:
                return False
        
        return True
    
    def detect_three_black_crows(self, data):
        """
        Detect a Three Black Crows pattern (bearish continuation).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 3 candles
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 3:
            return False
        
        # Get the three candles
        candles = [data.iloc[-3], data.iloc[-2], data.iloc[-1]]
        
        # All three candles should be bearish
        if not all(candle['close'] < candle['open'] for candle in candles):
            return False
        
        # Each candle should close lower than the previous
        if not (candles[1]['close'] < candles[0]['close'] and candles[2]['close'] < candles[1]['close']):
            return False
        
        # Each candle should open within the previous candle's body
        if not (candles[1]['open'] < candles[0]['open'] and 
                candles[1]['open'] > candles[0]['close'] and
                candles[2]['open'] < candles[1]['open'] and
                candles[2]['open'] > candles[1]['close']):
            return False
        
        # Check for small lower shadows (less than 15% of body size)
        for candle in candles:
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = abs(min(candle['open'], candle['close']) - candle['low'])
            
            if lower_shadow > 0.15 * body_size:
                return False
        
        return True
    
    def detect_rising_three(self, data):
        """
        Detect a Rising Three pattern (bullish continuation).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 5 candles
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 5:
            return False
        
        # Get the relevant candles
        first = data.iloc[-5]
        middle_three = [data.iloc[-4], data.iloc[-3], data.iloc[-2]]
        last = data.iloc[-1]
        
        # First candle should be bullish
        if first['close'] <= first['open']:
            return False
        
        # Last candle should be bullish
        if last['close'] <= last['open']:
            return False
        
        # Middle candles should be bearish
        if not all(candle['close'] < candle['open'] for candle in middle_three):
            return False
        
        # Middle candles should be contained within the range of the first candle
        for candle in middle_three:
            if candle['high'] > first['high'] or candle['low'] < first['low']:
                return False
        
        # Last candle should close above the first candle's close
        if last['close'] <= first['close']:
            return False
        
        return True
    
    def detect_falling_three(self, data):
        """
        Detect a Falling Three pattern (bearish continuation).
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data with at least 5 candles
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < 5:
            return False
        
        # Get the relevant candles
        first = data.iloc[-5]
        middle_three = [data.iloc[-4], data.iloc[-3], data.iloc[-2]]
        last = data.iloc[-1]
        
        # First candle should be bearish
        if first['close'] >= first['open']:
            return False
        
        # Last candle should be bearish
        if last['close'] >= last['open']:
            return False
        
        # Middle candles should be bullish
        if not all(candle['close'] > candle['open'] for candle in middle_three):
            return False
        
        # Middle candles should be contained within the range of the first candle
        for candle in middle_three:
            if candle['high'] > first['high'] or candle['low'] < first['low']:
                return False
        
        # Last candle should close below the first candle's close
        if last['close'] >= first['close']:
            return False
        
        return True
    
    def detect_entry_signal(self, df):
        """
        Detect entry signals based on triple candlestick patterns.
        
        This is a replacement for the existing method to focus on triple patterns.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        dict: Dictionary with entry signals and their patterns
        """
        signals = {}
        
        # Check for bullish patterns
        if self.detect_morning_star(df, doji_required=False):
            signals['morning_star'] = True
        
        if self.detect_morning_star(df, doji_required=True):
            signals['morning_doji_star'] = True
        
        if self.detect_three_white_soldiers(df):
            signals['three_white_soldiers'] = True
        
        if self.detect_rising_three(df):
            signals['rising_three'] = True
        
        # Check for bearish patterns
        if self.detect_evening_star(df, doji_required=False):
            signals['evening_star'] = True
        
        if self.detect_evening_star(df, doji_required=True):
            signals['evening_doji_star'] = True
        
        if self.detect_three_black_crows(df):
            signals['three_black_crows'] = True
        
        if self.detect_falling_three(df):
            signals['falling_three'] = True
        
        return signals
    
    def get_optimal_entry_price(self, df, pattern):
        """
        Get the optimal entry price based on the detected pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
        pattern: str
            Detected pattern
                
        Returns:
        --------
        float: Optimal entry price
        """
        if df is None or df.empty:
            return None
        
        last_candle = df.iloc[-1]
        
        # For bullish patterns, enter above the last candle's high
        if pattern in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three']:
            return last_candle['high'] * 1.001  # 0.1% above high
        
        # For bearish patterns, enter below the last candle's low
        elif pattern in ['evening_star', 'evening_doji_star', 'three_black_crows', 'falling_three']:
            return last_candle['low'] * 0.999  # 0.1% below low
        
        # Default entry at current price
        return last_candle['close']
    
    def get_optimal_stop_price(self, df, pattern):
        """
        Get the optimal stop loss price based on the detected pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
        pattern: str
            Detected pattern
                
        Returns:
        --------
        float: Optimal stop loss price
        """
        if df is None or df.empty:
            return None
        
        # For Morning Star patterns
        if pattern == 'morning_star' or pattern == 'morning_doji_star':
            # Use low of the middle candle as stop
            return df.iloc[-2]['low'] * 0.999  # 0.1% below middle candle low
        
        # For Three White Soldiers
        elif pattern == 'three_white_soldiers':
            # Use low of the last candle as stop
            return df.iloc[-1]['low'] * 0.999  # 0.1% below last candle low
        
        # For Rising Three
        elif pattern == 'rising_three':
            # Use low of the entire pattern as stop
            pattern_low = df.iloc[-5:]['low'].min()
            return pattern_low * 0.999  # 0.1% below pattern low
        
        # For Evening Star patterns
        elif pattern == 'evening_star' or pattern == 'evening_doji_star':
            # Use high of the middle candle as stop
            return df.iloc[-2]['high'] * 1.001  # 0.1% above middle candle high
        
        # For Three Black Crows
        elif pattern == 'three_black_crows':
            # Use high of the last candle as stop
            return df.iloc[-1]['high'] * 1.001  # 0.1% above last candle high
        
        # For Falling Three
        elif pattern == 'falling_three':
            # Use high of the entire pattern as stop
            pattern_high = df.iloc[-5:]['high'].max()
            return pattern_high * 1.001  # 0.1% above pattern high
        
        # Default stop based on last candle
        entry_price = self.get_optimal_entry_price(df, pattern)
        if pattern in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three']:
            return entry_price * 0.98  # 2% below entry for bullish patterns
        else:
            return entry_price * 1.02  # 2% above entry for bearish patterns
    
    def get_optimal_target_price(self, df, pattern):
        """
        Get the optimal target price based on detected pattern and risk-reward ratio.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
        pattern: str
            Detected pattern
            
        Returns:
        --------
        float: Optimal target price
        """
        if df is None or df.empty:
            return None
        
        # Get entry and stop prices
        entry_price = self.get_optimal_entry_price(df, pattern)
        stop_price = self.get_optimal_stop_price(df, pattern)
        
        if entry_price is None or stop_price is None:
            return None
        
        # Calculate risk
        risk = abs(entry_price - stop_price)
        
        # Use 2:1 reward-to-risk ratio
        if pattern in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three']:
            # For bullish patterns
            return entry_price + (risk * 2)
        else:
            # For bearish patterns
            return entry_price - (risk * 2)