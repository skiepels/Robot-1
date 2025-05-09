"""
Modified Candlestick Pattern Recognition Module

This module implements the detection of specific patterns:

1. Bull Flag: A continuation pattern formed after a strong uptrend, 
   consisting of a consolidation (flag) after a sharp move up (pole)

2. Bull Pennant: Similar to a flag but with converging trendlines forming
   a symmetrical triangle during the consolidation phase

3. Flat Top Breakout: A pattern where price consolidates with a flat resistance
   level on top, then breaks through this level on increased volume
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """
    Detects specific candlestick patterns for trade entry signals.
    
    This class contains methods to identify high-probability entry patterns
    in price data, focusing on Bull Flag, Bull Pennant, and Flat Top Breakout.
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
    
    def detect_bull_flag(self, data, lookback=10, min_pole_length_pct=5.0, max_flag_duration=7):
        """
        Detect a Bull Flag pattern.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data
        lookback: int
            Number of candles to look back for pattern formation
        min_pole_length_pct: float
            Minimum percentage increase required for the pole
        max_flag_duration: int
            Maximum number of candles allowed for the flag/consolidation
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < lookback:
            return False
        
        # Get the relevant candles
        recent_data = data.iloc[-lookback:]
        
        # Find potential pole (sharp move up)
        for i in range(2, min(7, len(recent_data) - 3)):
            # Potential pole start
            pole_start_idx = 0
            pole_end_idx = i
            
            pole_start_price = recent_data.iloc[pole_start_idx]['close']
            pole_end_price = recent_data.iloc[pole_end_idx]['close']
            
            # Calculate pole move percent
            pole_move_pct = (pole_end_price / pole_start_price - 1) * 100
            
            # Check if pole is strong enough
            if pole_move_pct < min_pole_length_pct:
                continue
            
            # Now check for consolidation/flag after the pole
            flag_start_idx = pole_end_idx
            flag_end_idx = len(recent_data) - 1
            
            # Ensure flag isn't too long
            if flag_end_idx - flag_start_idx > max_flag_duration:
                flag_end_idx = flag_start_idx + max_flag_duration
            
            flag_data = recent_data.iloc[flag_start_idx:flag_end_idx+1]
            
            # Criteria for a valid flag:
            # 1. Price should consolidate (not move too much in either direction)
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            
            # Calculate flag range as percentage of pole end price
            flag_range_pct = (flag_high - flag_low) / pole_end_price * 100
            
            # Flag range should be less than the pole's rise
            if flag_range_pct > pole_move_pct * 0.6:
                continue
            
            # 2. Flag should be at least 2 candles
            if len(flag_data) < 2:
                continue
            
            # 3. Final candle should show sign of upward movement
            final_candle = recent_data.iloc[-1]
            prev_candle = recent_data.iloc[-2]
            
            if not (final_candle['high'] > prev_candle['high'] and final_candle['close'] > final_candle['open']):
                continue
            
            # Check volume pattern (should decrease during flag, then increase on breakout)
            if 'volume' in recent_data.columns:
                # Average volume during pole
                pole_vol_avg = recent_data.iloc[pole_start_idx:pole_end_idx+1]['volume'].mean()
                
                # Average volume during early flag
                early_flag_end = min(flag_start_idx + 3, flag_end_idx)
                early_flag_vol_avg = recent_data.iloc[flag_start_idx:early_flag_end+1]['volume'].mean()
                
                # Volume on breakout candle
                breakout_vol = final_candle['volume']
                
                # Check if volume pattern matches expectations
                vol_decreasing_in_flag = early_flag_vol_avg < pole_vol_avg
                vol_increasing_on_breakout = breakout_vol > early_flag_vol_avg
                
                if not (vol_decreasing_in_flag and vol_increasing_on_breakout):
                    continue
            
            # Pattern detected
            logger.info(f"Bull Flag detected: Pole move: {pole_move_pct:.2f}%, Flag range: {flag_range_pct:.2f}%")
            return True
        
        return False
    
    def detect_bull_pennant(self, data, lookback=10, min_pole_length_pct=5.0, max_pennant_duration=7):
        """
        Detect a Bull Pennant pattern.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data
        lookback: int
            Number of candles to look back for pattern formation
        min_pole_length_pct: float
            Minimum percentage increase required for the pole
        max_pennant_duration: int
            Maximum number of candles allowed for the pennant/consolidation
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < lookback:
            return False
        
        # Get the relevant candles
        recent_data = data.iloc[-lookback:]
        
        # Find potential pole (sharp move up)
        for i in range(2, min(7, len(recent_data) - 4)):
            # Potential pole start
            pole_start_idx = 0
            pole_end_idx = i
            
            pole_start_price = recent_data.iloc[pole_start_idx]['close']
            pole_end_price = recent_data.iloc[pole_end_idx]['close']
            
            # Calculate pole move percent
            pole_move_pct = (pole_end_price / pole_start_price - 1) * 100
            
            # Check if pole is strong enough
            if pole_move_pct < min_pole_length_pct:
                continue
            
            # Now check for pennant pattern after the pole
            pennant_start_idx = pole_end_idx
            pennant_end_idx = len(recent_data) - 1
            
            # Ensure pennant isn't too long
            if pennant_end_idx - pennant_start_idx > max_pennant_duration:
                pennant_end_idx = pennant_start_idx + max_pennant_duration
            
            pennant_data = recent_data.iloc[pennant_start_idx:pennant_end_idx+1]
            
            if len(pennant_data) < 4:  # Need at least 4 candles to form a pennant
                continue
                
            # Calculate upper and lower trendlines for the pennant
            # For upper trendline, connect the highs
            # For lower trendline, connect the lows
            highs = pennant_data['high'].values
            lows = pennant_data['low'].values
            x = np.arange(len(pennant_data))
            
            try:
                # Fit upper trendline
                upper_slope, upper_intercept = np.polyfit(x, highs, 1)
                
                # Fit lower trendline
                lower_slope, lower_intercept = np.polyfit(x, lows, 1)
                
                # For a bull pennant, upper trendline should be descending (negative slope)
                # and lower trendline should be ascending (positive slope)
                if not (upper_slope < 0 and lower_slope > 0):
                    continue
                
                # Calculate convergence point
                # x at intersection: upper_intercept + upper_slope * x = lower_intercept + lower_slope * x
                # => upper_intercept - lower_intercept = (lower_slope - upper_slope) * x
                convergence_x = (upper_intercept - lower_intercept) / (lower_slope - upper_slope)
                
                # Convergence should be ahead (not too far but not already passed)
                if not (convergence_x > len(pennant_data) and convergence_x < len(pennant_data) * 2):
                    continue
                
                # Final candle should show breakout potential
                final_candle = recent_data.iloc[-1]
                prev_candle = recent_data.iloc[-2]
                
                if not (final_candle['high'] > prev_candle['high'] and final_candle['close'] > final_candle['open']):
                    continue
                
                # Check volume pattern (should decrease during pennant, then increase on breakout)
                if 'volume' in recent_data.columns:
                    # Average volume during pole
                    pole_vol_avg = recent_data.iloc[pole_start_idx:pole_end_idx+1]['volume'].mean()
                    
                    # Average volume during pennant
                    pennant_vol_avg = pennant_data['volume'].mean()
                    
                    # Volume on breakout candle
                    breakout_vol = final_candle['volume']
                    
                    # Check if volume pattern matches expectations
                    vol_decreasing_in_pennant = pennant_vol_avg < pole_vol_avg
                    vol_increasing_on_breakout = breakout_vol > pennant_vol_avg
                    
                    if not (vol_decreasing_in_pennant and vol_increasing_on_breakout):
                        continue
                
                # Pattern detected
                logger.info(f"Bull Pennant detected: Pole move: {pole_move_pct:.2f}%")
                return True
                
            except np.linalg.LinAlgError:
                # If we can't fit the trendlines, this isn't a valid pennant
                continue
        
        return False
    
    def detect_flat_top_breakout(self, data, lookback=10, min_touches=2, max_deviation_pct=0.5):
        """
        Detect a Flat Top Breakout pattern.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            OHLCV data
        lookback: int
            Number of candles to look back for pattern formation
        min_touches: int
            Minimum number of times price should touch the resistance level
        max_deviation_pct: float
            Maximum percentage deviation allowed in the resistance level
            
        Returns:
        --------
        bool: True if the pattern is detected
        """
        if len(data) < lookback:
            return False
        
        # Get the relevant candles
        recent_data = data.iloc[-lookback:]
        
        # Check the most recent candle for breakout characteristics
        current_candle = recent_data.iloc[-1]
        prev_candle = recent_data.iloc[-2]
        
        # Current candle should be bullish
        if not self.is_bullish_candle(current_candle):
            return False
        
        # Current candle should close above the previous candle's high
        if current_candle['close'] <= prev_candle['high']:
            return False
        
        # Now let's identify potential flat top (resistance level)
        # Get data excluding the current (breakout) candle
        consolidation_data = recent_data.iloc[:-1]
        
        if len(consolidation_data) < 4:  # Need at least 4 candles to identify a flat top
            return False
        
        # Find potential resistance levels using the highs
        highs = consolidation_data['high'].values
        
        # Use the highest high as a starting point for resistance
        potential_resistance = np.max(highs)
        
        # Calculate number of candles that come within a small percentage of the resistance
        touches = sum(1 for h in highs if abs(h - potential_resistance) / potential_resistance * 100 <= max_deviation_pct)
        
        if touches < min_touches:
            return False
        
        # Check volume on breakout candle
        if 'volume' in data.columns:
            # Get average volume during consolidation
            avg_vol = consolidation_data['volume'].mean()
            
            # Check if breakout candle volume is higher
            if current_candle['volume'] <= avg_vol:
                return False
        
        # Pattern detected
        logger.info(f"Flat Top Breakout detected: Resistance: {potential_resistance:.2f}, Touches: {touches}")
        return True
    
    def detect_entry_signal(self, df):
        """
        Detect entry signals based on the three defined patterns.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        dict: Dictionary with entry signals and their patterns
        """
        signals = {}
        
        # Check for Bull Flag
        if self.detect_bull_flag(df):
            signals['bull_flag'] = True
        
        # Check for Bull Pennant
        if self.detect_bull_pennant(df):
            signals['bull_pennant'] = True
        
        # Check for Flat Top Breakout
        if self.detect_flat_top_breakout(df):
            signals['flat_top_breakout'] = True
        
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
        if df is None or df.empty or len(df) < 2:
            return None
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        if pattern == 'bull_flag' or pattern == 'bull_pennant':
            # For flags and pennants, enter above the last candle's high
            return last_candle['high'] * 1.001  # Slight buffer for breakout
        
        elif pattern == 'flat_top_breakout':
            # For flat top breakout, find the resistance level
            consolidation_data = df.iloc[:-1]
            resistance = consolidation_data['high'].max()
            
            # Enter slightly above the resistance level
            return resistance * 1.005  # 0.5% above resistance
        
        # Default entry
        return last_candle['high'] * 1.001
    
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
        if df is None or df.empty or len(df) < 3:
            return None
        
        last_candle = df.iloc[-1]
        
        if pattern == 'bull_flag':
            # For bull flag, use the low of the flag as stop
            # Find where the flag starts (after a strong uptrend)
            lookback = min(10, len(df) - 1)
            recent_data = df.iloc[-lookback:]
            
            # Find potential pole end (where consolidation begins)
            # A simple approach: find where the highest close is, then take one candle after that
            max_close_idx = recent_data['close'].idxmax()
            max_close_position = recent_data.index.get_loc(max_close_idx)
            
            flag_start_position = min(max_close_position + 1, len(recent_data) - 2)
            flag_data = recent_data.iloc[flag_start_position:-1]  # Exclude breakout candle
            
            # Use the lowest low of the flag
            flag_low = flag_data['low'].min()
            return flag_low * 0.995  # Slight buffer for stop
            
        elif pattern == 'bull_pennant':
            # For bull pennant, use the lower trendline as stop
            lookback = min(10, len(df))
            pennant_data = df.iloc[-lookback:]
            
            # Estimate the lower trendline using the lows
            recent_lows = pennant_data['low'].values
            x = np.arange(len(pennant_data))
            
            try:
                # Fit lower trendline
                slope, intercept = np.polyfit(x, recent_lows, 1)
                
                # Get the most recent value of the trendline
                latest_trendline_value = slope * (len(pennant_data) - 1) + intercept
                
                # Use a value slightly below the trendline
                return latest_trendline_value * 0.99
            except np.linalg.LinAlgError:
                # If we can't fit the trendline, use the lowest low of recent candles
                return pennant_data['low'].min() * 0.995
                
        elif pattern == 'flat_top_breakout':
            # For flat top breakout, use the previous resistance (now support) as stop
            consolidation_data = df.iloc[:-1]
            resistance = consolidation_data['high'].max()
            
            # Stop slightly below the previous resistance
            return resistance * 0.99
        
        # Default stop - use recent swing low
        return df.iloc[-5:]['low'].min() * 0.995
    
    def get_optimal_target_price(self, df, pattern):
        """
        Get the optimal target price based on the detected pattern.
        
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
        # Get entry and stop prices
        entry_price = self.get_optimal_entry_price(df, pattern)
        stop_price = self.get_optimal_stop_price(df, pattern)
        
        if entry_price is None or stop_price is None:
            return None
        
        # Calculate risk
        risk = entry_price - stop_price
        
        if pattern == 'bull_flag':
            # For bull flag, measure the pole and project it from the breakout
            lookback = min(10, len(df) - 1)
            recent_data = df.iloc[-lookback:]
            
            # Find the pole (strong upward move before consolidation)
            # A simple approach: find the lowest low in the first few candles, then the highest high before consolidation
            pole_low = recent_data.iloc[:3]['low'].min()
            
            # Find where consolidation begins (after the highest high in the first part)
            high_points = recent_data.iloc[:5]['high']
            pole_high = high_points.max()
            
            # Measure the pole height
            pole_height = pole_high - pole_low
            
            # Project the pole height from the breakout point
            return entry_price + pole_height
            
        elif pattern == 'bull_pennant':
            # For bull pennant, similar to flag - measure the pole and project it
            lookback = min(10, len(df) - 1)
            recent_data = df.iloc[-lookback:]
            
            # Find the pole (strong upward move before the pennant)
            pole_low = recent_data.iloc[:3]['low'].min()
            high_points = recent_data.iloc[:5]['high']
            pole_high = high_points.max()
            
            # Measure the pole height
            pole_height = pole_high - pole_low
            
            # Project the pole height from the breakout point
            return entry_price + pole_height
            
        elif pattern == 'flat_top_breakout':
            # For flat top breakout, use the consolidated range and project it upward
            lookback = min(10, len(df) - 1)
            consolidation_data = df.iloc[-lookback:-1]  # Exclude breakout candle
            
            # Measure the consolidation range
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Project the range from the breakout point (often gives a minimum target)
            return entry_price + consolidation_range
        
        # Default target: Use 2:1 reward-to-risk ratio
        return entry_price + (risk * 2)