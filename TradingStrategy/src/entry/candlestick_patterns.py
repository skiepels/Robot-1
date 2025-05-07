"""
Candlestick Pattern Recognition Module

This module implements the candlestick pattern detection logic for various entry patterns:
- Bull Flag
- Micro Pullback
- New High Breakout
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """
    Detects various candlestick patterns for trade entry signals.
    
    This class contains methods to identify high-probability entry patterns
    commonly used in momentum day trading strategies.
    """
    
    def __init__(self):
        """Initialize the candlestick pattern detector."""
        pass
    
    def is_bullish_candle(self, open_price, close_price):
        """
        Determine if a candle is bullish (close > open).
        
        Parameters:
        -----------
        open_price: float
            Opening price
        close_price: float
            Closing price
            
        Returns:
        --------
        bool: True if bullish, False otherwise
        """
        return close_price > open_price
    
    def is_bearish_candle(self, open_price, close_price):
        """
        Determine if a candle is bearish (close < open).
        
        Parameters:
        -----------
        open_price: float
            Opening price
        close_price: float
            Closing price
            
        Returns:
        --------
        bool: True if bearish, False otherwise
        """
        return close_price < open_price
    
    def calculate_body_size(self, open_price, close_price):
        """
        Calculate the absolute size of the candle body.
        
        Parameters:
        -----------
        open_price: float
            Opening price
        close_price: float
            Closing price
            
        Returns:
        --------
        float: Absolute size of the candle body
        """
        return abs(close_price - open_price)
    
    def calculate_upper_wick(self, open_price, close_price, high_price):
        """
        Calculate the size of the upper wick/shadow.
        
        Parameters:
        -----------
        open_price: float
            Opening price
        close_price: float
            Closing price
        high_price: float
            High price
            
        Returns:
        --------
        float: Size of the upper wick
        """
        return high_price - max(open_price, close_price)
    
    def calculate_lower_wick(self, open_price, close_price, low_price):
        """
        Calculate the size of the lower wick/shadow.
        
        Parameters:
        -----------
        open_price: float
            Opening price
        close_price: float
            Closing price
        low_price: float
            Low price
            
        Returns:
        --------
        float: Size of the lower wick
        """
        return min(open_price, close_price) - low_price
    
    def is_bull_flag(self, df):
        """
        Detect a Bull Flag pattern in the price data.
        
        A bull flag consists of a strong upward move (the pole)
        followed by a consolidation period with lower highs (the flag).
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data with at least 5-10 candles
            
        Returns:
        --------
        bool: True if a bull flag pattern is detected
        """
        if df is None or len(df) < 7:
            return False
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Get the most recent 7 candles
            recent_data = data.iloc[-7:]
            
            # Check for a strong upward move in the first 3 candles (the pole)
            first_3_candles = recent_data.iloc[:3]
            pole_start_price = first_3_candles['open'].iloc[0]
            pole_end_price = first_3_candles['close'].iloc[-1]
            
            # Calculate pole percentage gain
            pole_gain_pct = (pole_end_price / pole_start_price - 1) * 100
            
            # The pole should show a significant increase (at least 3%)
            if pole_gain_pct < 3:
                return False
            
            # Check for consolidation with lower highs (the flag)
            flag_candles = recent_data.iloc[3:]
            
            # Check for at least 2 lower highs
            previous_high = first_3_candles['high'].max()
            lower_highs_count = 0
            
            for i in range(len(flag_candles) - 1):
                current_high = flag_candles['high'].iloc[i]
                
                if current_high < previous_high:
                    lower_highs_count += 1
                
                previous_high = current_high
            
            # Need at least 2 lower highs to confirm the flag
            if lower_highs_count < 2:
                return False
            
            # Check volume pattern - should decrease during flag
            if 'volume' in flag_candles.columns:
                first_half_vol = flag_candles['volume'].iloc[:len(flag_candles)//2].mean()
                second_half_vol = flag_candles['volume'].iloc[len(flag_candles)//2:].mean()
                
                # Volume should decrease in the second half of the flag
                if second_half_vol >= first_half_vol:
                    return False
            
            # Check for breakout in the last candle
            last_candle = recent_data.iloc[-1]
            flag_high = recent_data.iloc[3:-1]['high'].max()
            
            # Last candle should be bullish
            if not self.is_bullish_candle(last_candle['open'], last_candle['close']):
                return False
            
            # Last candle should break above the flag high
            if last_candle['close'] <= flag_high:
                return False
            
            # All criteria met, bull flag confirmed
            return True
            
        except Exception as e:
            logger.error(f"Error detecting bull flag: {e}")
            return False
    
    def is_micro_pullback(self, df):
        """
        Detect a Micro Pullback pattern in the price data.
        
        A micro pullback consists of a strong move up, followed by
        a small retracement (red candle or candle with long lower wick),
        then a breakout above the pullback high.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data with at least 4-5 candles
            
        Returns:
        --------
        bool: True if a micro pullback pattern is detected
        """
        if df is None or len(df) < 5:
            return False
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Get the most recent 5 candles
            recent_data = data.iloc[-5:]
            
            # Check for uptrend in the first 3 candles
            first_3_candles = recent_data.iloc[:3]
            uptrend = first_3_candles['close'].iloc[-1] > first_3_candles['close'].iloc[0]
            
            if not uptrend:
                return False
            
            # Check for micro pullback in the 4th candle
            pullback_candle = recent_data.iloc[-2]
            
            # Pullback criteria: either a red candle or a candle with a significant lower wick
            is_red = pullback_candle['close'] < pullback_candle['open']
            
            lower_wick = self.calculate_lower_wick(
                pullback_candle['open'], 
                pullback_candle['close'], 
                pullback_candle['low']
            )
            
            body_size = self.calculate_body_size(pullback_candle['open'], pullback_candle['close'])
            has_long_lower_wick = lower_wick > body_size * 0.5
            
            if not (is_red or has_long_lower_wick):
                return False
            
            # Check for breakout in the last candle
            last_candle = recent_data.iloc[-1]
            
            # Last candle should be bullish
            if not self.is_bullish_candle(last_candle['open'], last_candle['close']):
                return False
            
            # Last candle should break above the pullback candle's high
            if last_candle['high'] <= pullback_candle['high']:
                return False
            
            # All criteria met, micro pullback confirmed
            return True
            
        except Exception as e:
            logger.error(f"Error detecting micro pullback: {e}")
            return False
    
    def is_new_high_breakout(self, df):
        """
        Detect a "First Candle to Make a New High" pattern.
        
        This pattern occurs when a stock makes a new high after 
        a period of consolidation or pullback.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data with at least 5-7 candles
            
        Returns:
        --------
        bool: True if a new high breakout pattern is detected
        """
        if df is None or len(df) < 6:
            return False
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Get the most recent 6 candles
            recent_data = data.iloc[-6:]
            
            # Find the previous high in the lookback period (excluding most recent candle)
            previous_high = recent_data.iloc[:-1]['high'].max()
            
            # Get the last candle
            last_candle = recent_data.iloc[-1]
            
            # Last candle should be bullish
            if not self.is_bullish_candle(last_candle['open'], last_candle['close']):
                return False
            
            # Last candle should make a new high
            if last_candle['high'] <= previous_high:
                return False
            
            # Check for prior consolidation or pullback
            # (at least 2-3 candles without making new highs)
            prior_candles = recent_data.iloc[1:-1]  # Skip the first and last candles
            
            new_high_count = 0
            prior_high = recent_data.iloc[0]['high']
            
            for i in range(len(prior_candles)):
                if prior_candles['high'].iloc[i] > prior_high:
                    new_high_count += 1
                
                prior_high = max(prior_high, prior_candles['high'].iloc[i])
            
            # Should have no more than 1 new high in the prior candles
            # to confirm consolidation/pullback before breakout
            if new_high_count > 1:
                return False
            
            # Check if the volume on the breakout candle is higher
            if 'volume' in recent_data.columns:
                avg_volume = recent_data.iloc[:-1]['volume'].mean()
                breakout_volume = last_candle['volume']
                
                # Breakout volume should be higher than average
                if breakout_volume <= avg_volume:
                    return False
            
            # All criteria met, new high breakout confirmed
            return True
            
        except Exception as e:
            logger.error(f"Error detecting new high breakout: {e}")
            return False
    
    def detect_entry_signal(self, df):
        """
        Detect all potential entry signals in the price data.
        
        This method checks for bull flag, micro pullback, and new high breakout patterns.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        dict: Dictionary with entry signals and their corresponding patterns
        """
        signals = {}
        
        # Check for bull flag
        if self.is_bull_flag(df):
            signals['bull_flag'] = True
        
        # Check for micro pullback
        if self.is_micro_pullback(df):
            signals['micro_pullback'] = True
        
        # Check for new high breakout
        if self.is_new_high_breakout(df):
            signals['new_high_breakout'] = True
        
        return signals
    
    def get_optimal_entry_price(self, df, pattern):
        """
        Get the optimal entry price based on the detected pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
        pattern: str
            Detected pattern (bull_flag, micro_pullback, new_high_breakout)
            
        Returns:
        --------
        float: Optimal entry price
        """
        if df is None or df.empty:
            return None
        
        last_candle = df.iloc[-1]
        
        # Default entry at current price
        entry_price = last_candle['close']
        
        # For bull flag, entry is above the flag high
        if pattern == 'bull_flag':
            flag_high = df.iloc[-5:-1]['high'].max()
            entry_price = max(last_candle['close'], flag_high)
        
        # For micro pullback, entry is above the pullback candle's high
        elif pattern == 'micro_pullback':
            pullback_candle = df.iloc[-2]
            entry_price = max(last_candle['close'], pullback_candle['high'])
        
        # For new high breakout, entry is the breakout price
        elif pattern == 'new_high_breakout':
            entry_price = last_candle['close']
        
        return entry_price
    
    def get_optimal_stop_price(self, df, pattern):
        """
        Get the optimal stop loss price based on the detected pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
        pattern: str
            Detected pattern (bull_flag, micro_pullback, new_high_breakout)
            
        Returns:
        --------
        float: Optimal stop loss price
        """
        if df is None or df.empty:
            return None
        
        last_candle = df.iloc[-1]
        entry_price = self.get_optimal_entry_price(df, pattern)
        
        # Default stop below the last candle's low
        stop_price = last_candle['low']
        
        # For bull flag, stop below the flag low
        if pattern == 'bull_flag':
            flag_low = df.iloc[-5:-1]['low'].min()
            stop_price = flag_low
        
        # For micro pullback, stop below the pullback candle's low
        elif pattern == 'micro_pullback':
            pullback_candle = df.iloc[-2]
            stop_price = pullback_candle['low']
        
        # For new high breakout, stop below the last candle's low
        elif pattern == 'new_high_breakout':
            stop_price = last_candle['low']
        
        # Ensure stop is not too tight (at least 2% below entry)
        min_stop = entry_price * 0.98
        if stop_price > min_stop:
            stop_price = min_stop
        
        return stop_price
    
    def get_optimal_target_price(self, entry_price, stop_price, profit_loss_ratio=2.0):
        """
        Get the optimal profit target price based on risk-reward ratio.
        
        Parameters:
        -----------
        entry_price: float
            Entry price
        stop_price: float
            Stop loss price
        profit_loss_ratio: float
            Desired profit-to-loss ratio
            
        Returns:
        --------
        float: Optimal target price
        """
        if entry_price is None or stop_price is None:
            return None
        
        # Calculate risk per share
        risk = entry_price - stop_price
        
        # Calculate target based on risk-reward ratio
        target_price = entry_price + (risk * profit_loss_ratio)
        
        return target_price