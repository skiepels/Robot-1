"""
Candlestick Pattern Recognition Module

This module implements the candlestick pattern recognition logic based on 
Ross Cameron's bull flag and other technical patterns used for day trading.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    def __init__(self):
        """Initialize the candlestick pattern detector"""
        pass

    def is_bullish_candle(self, open_price, close_price):
        """Determine if a candle is bullish (close > open)"""
        return close_price > open_price
    
    def is_bearish_candle(self, open_price, close_price):
        """Determine if a candle is bearish (close < open)"""
        return close_price < open_price
    
    def calculate_body_size(self, open_price, close_price):
        """Calculate the absolute size of the candle body"""
        return abs(close_price - open_price)
    
    def calculate_upper_wick(self, open_price, close_price, high_price):
        """Calculate the size of the upper wick/shadow"""
        return high_price - max(open_price, close_price)
    
    def calculate_lower_wick(self, open_price, close_price, low_price):
        """Calculate the size of the lower wick/shadow"""
        return min(open_price, close_price) - low_price
    
    def detect_hammer(self, df, idx):
        """
        Detect a Hammer pattern
        - Small body at the upper end of the trading range
        - Little or no upper shadow
        - Long lower shadow (at least 2x the body)
        - Appears in a downtrend
        """
        if idx < 3 or idx >= len(df):
            return False
            
        # Check for downtrend
        if not (df['close'].iloc[idx-3:idx].mean() > df['close'].iloc[idx]):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        body_size = self.calculate_body_size(open_price, close_price)
        upper_wick = self.calculate_upper_wick(open_price, close_price, high_price)
        lower_wick = self.calculate_lower_wick(open_price, close_price, low_price)
        
        # Small upper wick
        if upper_wick > body_size * 0.3:
            return False
            
        # Long lower wick
        if lower_wick < body_size * 2:
            return False
            
        return True
    
    def detect_inverted_hammer(self, df, idx):
        """
        Detect an Inverted Hammer pattern
        - Small body at the lower end of the trading range
        - Little or no lower shadow
        - Long upper shadow (at least 2x the body)
        - Appears in a downtrend
        """
        if idx < 3 or idx >= len(df):
            return False
            
        # Check for downtrend
        if not (df['close'].iloc[idx-3:idx].mean() > df['close'].iloc[idx]):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        body_size = self.calculate_body_size(open_price, close_price)
        upper_wick = self.calculate_upper_wick(open_price, close_price, high_price)
        lower_wick = self.calculate_lower_wick(open_price, close_price, low_price)
        
        # Long upper wick
        if upper_wick < body_size * 2:
            return False
            
        # Small lower wick
        if lower_wick > body_size * 0.3:
            return False
            
        return True
    
    def detect_dragonfly_doji(self, df, idx):
        """
        Detect a Dragonfly Doji pattern
        - Open and close are at or near the high
        - Long lower shadow
        """
        if idx >= len(df):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        total_range = high_price - low_price
        body_size = self.calculate_body_size(open_price, close_price)
        upper_wick = self.calculate_upper_wick(open_price, close_price, high_price)
        
        # Body should be very small
        if body_size > total_range * 0.1:
            return False
            
        # Upper wick should be very small or non-existent
        if upper_wick > total_range * 0.1:
            return False
            
        # Open and close should be near the high
        return (high_price - open_price <= total_range * 0.1 and 
                high_price - close_price <= total_range * 0.1)
    
    def detect_doji(self, df, idx):
        """
        Detect a Doji pattern
        - Very small body (open and close are very close)
        - Can have upper and lower shadows
        """
        if idx >= len(df):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        total_range = high_price - low_price
        body_size = self.calculate_body_size(open_price, close_price)
        
        # Body should be very small compared to the total range
        return body_size <= total_range * 0.1
    
    def detect_shooting_star(self, df, idx):
        """
        Detect a Shooting Star pattern
        - Small body at the lower end of the trading range
        - Little or no lower shadow
        - Long upper shadow (at least 2x the body)
        - Appears in an uptrend
        """
        if idx < 3 or idx >= len(df):
            return False
            
        # Check for uptrend
        if not (df['close'].iloc[idx-3:idx].mean() < df['close'].iloc[idx]):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        body_size = self.calculate_body_size(open_price, close_price)
        upper_wick = self.calculate_upper_wick(open_price, close_price, high_price)
        lower_wick = self.calculate_lower_wick(open_price, close_price, low_price)
        
        # Long upper wick
        if upper_wick < body_size * 2:
            return False
            
        # Small lower wick
        if lower_wick > body_size * 0.3:
            return False
            
        return True
    
    def detect_hanging_man(self, df, idx):
        """
        Detect a Hanging Man pattern
        - Small body at the upper end of the trading range
        - Little or no upper shadow
        - Long lower shadow (at least 2x the body)
        - Appears in an uptrend
        """
        if idx < 3 or idx >= len(df):
            return False
            
        # Check for uptrend
        if not (df['close'].iloc[idx-3:idx].mean() < df['close'].iloc[idx]):
            return False
            
        open_price = df['open'].iloc[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]
        
        body_size = self.calculate_body_size(open_price, close_price)
        upper_wick = self.calculate_upper_wick(open_price, close_price, high_price)
        lower_wick = self.calculate_lower_wick(open_price, close_price, low_price)
        
        # Small upper wick
        if upper_wick > body_size * 0.3:
            return False
            
        # Long lower wick
        if lower_wick < body_size * 2:
            return False
            
        return True
    
    def detect_bullish_engulfing(self, df, idx):
        """
        Detect a Bullish Engulfing pattern
        - First candle is bearish (red)
        - Second candle is bullish (green) and completely engulfs the first
        """
        if idx < 1 or idx >= len(df):
            return False
            
        prev_open = df['open'].iloc[idx-1]
        prev_close = df['close'].iloc[idx-1]
        curr_open = df['open'].iloc[idx]
        curr_close = df['close'].iloc[idx]
        
        # First candle is bearish
        if not self.is_bearish_candle(prev_open, prev_close):
            return False
            
        # Second candle is bullish
        if not self.is_bullish_candle(curr_open, curr_close):
            return False
            
        # Second candle engulfs the first
        return curr_open <= prev_close and curr_close >= prev_open
    
    def detect_bearish_engulfing(self, df, idx):
        """
        Detect a Bearish Engulfing pattern
        - First candle is bullish (green)
        - Second candle is bearish (red) and completely engulfs the first
        """
        if idx < 1 or idx >= len(df):
            return False
            
        prev_open = df['open'].iloc[idx-1]
        prev_close = df['close'].iloc[idx-1]
        curr_open = df['open'].iloc[idx]
        curr_close = df['close'].iloc[idx]
        
        # First candle is bullish
        if not self.is_bullish_candle(prev_open, prev_close):
            return False
            
        # Second candle is bearish
        if not self.is_bearish_candle(curr_open, curr_close):
            return False
            
        # Second candle engulfs the first
        return curr_open >= prev_close and curr_close <= prev_open
    
    def detect_morning_star(self, df, idx):
        """
        Detect a Morning Star pattern
        - First candle is bearish
        - Second candle is a small body (often a doji)
        - Third candle is bullish and closes well into the first candle
        """
        if idx < 2 or idx >= len(df):
            return False
            
        # Check the three candles
        first_open = df['open'].iloc[idx-2]
        first_close = df['close'].iloc[idx-2]
        second_open = df['open'].iloc[idx-1]
        second_close = df['close'].iloc[idx-1]
        third_open = df['open'].iloc[idx]
        third_close = df['close'].iloc[idx]
        
        # First candle is bearish
        if not self.is_bearish_candle(first_open, first_close):
            return False
            
        # Third candle is bullish
        if not self.is_bullish_candle(third_open, third_close):
            return False
            
        # Second candle has a small body
        second_body = self.calculate_body_size(second_open, second_close)
        first_body = self.calculate_body_size(first_open, first_close)
        third_body = self.calculate_body_size(third_open, third_close)
        
        if second_body >= min(first_body, third_body) * 0.5:
            return False
            
        # Third candle closes well into the first candle's body
        midpoint_first = first_open + (first_close - first_open) * 0.5
        
        return third_close > midpoint_first
    
    def detect_evening_star(self, df, idx):
        """
        Detect an Evening Star pattern
        - First candle is bullish
        - Second candle is a small body (often a doji)
        - Third candle is bearish and closes well into the first candle
        """
        if idx < 2 or idx >= len(df):
            return False
            
        # Check the three candles
        first_open = df['open'].iloc[idx-2]
        first_close = df['close'].iloc[idx-2]
        second_open = df['open'].iloc[idx-1]
        second_close = df['close'].iloc[idx-1]
        third_open = df['open'].iloc[idx]
        third_close = df['close'].iloc[idx]
        
        # First candle is bullish
        if not self.is_bullish_candle(first_open, first_close):
            return False
            
        # Third candle is bearish
        if not self.is_bearish_candle(third_open, third_close):
            return False
            
        # Second candle has a small body
        second_body = self.calculate_body_size(second_open, second_close)
        first_body = self.calculate_body_size(first_open, first_close)
        third_body = self.calculate_body_size(third_open, third_close)
        
        if second_body >= min(first_body, third_body) * 0.5:
            return False
            
        # Third candle closes well into the first candle's body
        midpoint_first = first_close - (first_close - first_open) * 0.5
        
        return third_close < midpoint_first
    
    def detect_three_white_soldiers(self, df, idx):
        """
        Detect a Three White Soldiers pattern
        - Three consecutive bullish candles
        - Each candle opens within the previous candle's body
        - Each candle closes higher than the previous candle
        - Each candle has small upper wicks
        """
        if idx < 2 or idx >= len(df):
            return False
            
        # Check the three candles
        candles = []
        for i in range(3):
            candle_idx = idx - 2 + i
            open_price = df['open'].iloc[candle_idx]
            close_price = df['close'].iloc[candle_idx]
            high_price = df['high'].iloc[candle_idx]
            
            # Check if bullish
            if not self.is_bullish_candle(open_price, close_price):
                return False
                
            # Store for further checks
            candles.append({
                'open': open_price,
                'close': close_price,
                'high': high_price
            })
        
        # Each candle opens within the previous candle's body
        for i in range(1, 3):
            if not (candles[i]['open'] > candles[i-1]['open'] and 
                   candles[i]['open'] < candles[i-1]['close']):
                return False
        
        # Each candle closes higher than the previous
        for i in range(1, 3):
            if candles[i]['close'] <= candles[i-1]['close']:
                return False
        
        # Each candle has small upper wicks
        for candle in candles:
            body_size = candle['close'] - candle['open']
            upper_wick = candle['high'] - candle['close']
            
            if upper_wick > body_size * 0.3:
                return False
        
        return True
    
    def detect_three_black_crows(self, df, idx):
        """
        Detect a Three Black Crows pattern
        - Three consecutive bearish candles
        - Each candle opens within the previous candle's body
        - Each candle closes lower than the previous candle
        - Each candle has small lower wicks
        """
        if idx < 2 or idx >= len(df):
            return False
            
        # Check the three candles
        candles = []
        for i in range(3):
            candle_idx = idx - 2 + i
            open_price = df['open'].iloc[candle_idx]
            close_price = df['close'].iloc[candle_idx]
            low_price = df['low'].iloc[candle_idx]
            
            # Check if bearish
            if not self.is_bearish_candle(open_price, close_price):
                return False
                
            # Store for further checks
            candles.append({
                'open': open_price,
                'close': close_price,
                'low': low_price
            })
        
        # Each candle opens within the previous candle's body
        for i in range(1, 3):
            if not (candles[i]['open'] < candles[i-1]['open'] and 
                   candles[i]['open'] > candles[i-1]['close']):
                return False
        
        # Each candle closes lower than the previous
        for i in range(1, 3):
            if candles[i]['close'] >= candles[i-1]['close']:
                return False
        
        # Each candle has small lower wicks
        for candle in candles:
            body_size = candle['open'] - candle['close']
            lower_wick = candle['close'] - candle['low']
            
            if lower_wick > body_size * 0.3:
                return False
        
        return True
    
    def detect_bull_flag(self, df, lookback=7):
        """
        Detect a Bull Flag pattern as described by Ross Cameron:
        1. Strong upward move (the "pole")
        2. Series of lower highs and lower lows (the "flag")
        3. Volume decreases during the flag formation
        4. Flag should hold above 50% retrace of the pole
        
        Returns a list of indices where the pattern completes
        """
        if len(df) < lookback + 2:
            return []
        
        bull_flags = []
        
        # Look for potential bull flags
        for i in range(lookback + 1, len(df)):
            # Check for a strong upward move before the potential flag
            pole_start_idx = i - lookback - 1
            pole_end_idx = i - lookback
            
            # Ensure there's enough data
            if pole_start_idx < 0:
                continue
                
            pole_start_price = df['close'].iloc[pole_start_idx]
            pole_end_price = df['close'].iloc[pole_end_idx]
            
            # The pole should show a significant increase (at least 5%)
            pole_pct_change = (pole_end_price / pole_start_price - 1) * 100
            if pole_pct_change < 5:
                continue
                
            # Check for flag pattern (consolidation with lower highs)
            flag_start_idx = pole_end_idx
            flag_end_idx = i
            
            flag_data = df.iloc[flag_start_idx:flag_end_idx+1]
            
            # Flag should have at least 3 candles
            if len(flag_data) < 3:
                continue
                
            # Flag should show lower highs
            has_lower_highs = True
            for j in range(2, len(flag_data)):
                if flag_data['high'].iloc[j] > flag_data['high'].iloc[j-1]:
                    has_lower_highs = False
                    break
                    
            if not has_lower_highs:
                continue
                
            # Flag should hold above 50% retracement of the pole
            pole_range = pole_end_price - pole_start_price
            retracement_level = pole_end_price - pole_range * 0.5
            
            if flag_data['low'].min() < retracement_level:
                continue
                
            # Volume should decrease during flag formation
            # Check if at least the second half has lower volume than first half
            if 'volume' in df.columns and len(flag_data) >= 4:
                first_half_vol = flag_data['volume'].iloc[:len(flag_data)//2].mean()
                second_half_vol = flag_data['volume'].iloc[len(flag_data)//2:].mean()
                
                if second_half_vol >= first_half_vol:
                    continue
                    
            # Pattern is complete, add to results
            bull_flags.append(i)
        
        return bull_flags
    
    def detect_first_candle_to_make_new_high(self, df, lookback=5):
        """
        Detect the "first candle to make a new high" pattern after a pullback.
        This is Ross Cameron's key entry strategy.
        
        Returns a list of indices where the pattern occurs.
        """
        if len(df) < lookback + 2:
            return []
        
        new_high_breakouts = []
        
        # Look for potential breakouts
        for i in range(lookback + 2, len(df)):
            # Find the high before the pullback
            pre_pullback_high = df['high'].iloc[i-lookback:i-1].max()
            
            # Check if current candle makes a new high
            current_high = df['high'].iloc[i]
            
            if current_high <= pre_pullback_high:
                continue
                
            # Check if previous candle did not make a new high
            prev_high = df['high'].iloc[i-1]
            
            if prev_high >= pre_pullback_high:
                continue
                
            # Check for a pullback before the breakout
            # We need at least 2 consecutive lower highs
            has_pullback = False
            for j in range(i-3, i-1):
                if j < 0:
                    continue
                    
                if df['high'].iloc[j] < df['high'].iloc[j-1]:
                    has_pullback = True
                    break
                    
            if not has_pullback:
                continue
                
            # Current candle should be bullish
            if not self.is_bullish_candle(df['open'].iloc[i], df['close'].iloc[i]):
                continue
                
            # Pattern is complete, add to results
            new_high_breakouts.append(i)
        
        return new_high_breakouts
    
    def detect_micro_pullback(self, df, lookback=3):
        """
        Detect Micro Pullback patterns as described by Ross Cameron:
        1. Stock is moving up quickly
        2. Very small red candle or a candle with lower wick
        3. Next candle breaks above the previous candle's high
        
        Returns a list of indices where the pattern occurs.
        """
        if len(df) < lookback + 1:
            return []
        
        micro_pullbacks = []
        
        # Look for potential micro pullbacks
        for i in range(lookback, len(df) - 1):  # -1 because we need the next candle too
            # Check if stock was moving up before this
            prev_candles = df.iloc[i-lookback:i]
            
            # Need at least 2 out of 3 previous candles to be bullish
            bullish_count = sum(1 for j in range(len(prev_candles)) 
                               if self.is_bullish_candle(prev_candles['open'].iloc[j], 
                                                        prev_candles['close'].iloc[j]))
            
            if bullish_count < 2:
                continue
                
            # Current candle is either a small red or has a significant lower wick
            curr_open = df['open'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_high = df['high'].iloc[i]
            
            is_small_red = (self.is_bearish_candle(curr_open, curr_close) and 
                          self.calculate_body_size(curr_open, curr_close) < 
                          (prev_candles['high'].max() - prev_candles['low'].min()) * 0.2)
            
            has_lower_wick = (curr_low < min(curr_open, curr_close) and 
                            (min(curr_open, curr_close) - curr_low) > 
                            self.calculate_body_size(curr_open, curr_close) * 0.5)
            
            if not (is_small_red or has_lower_wick):
                continue
                
            # Next candle should break above the current candle's high
            next_high = df['high'].iloc[i+1]
            
            if next_high <= curr_high:
                continue
                
            # Pattern is complete, add to results
            micro_pullbacks.append(i)
        
        return micro_pullbacks
    
    def detect_tweezer_bottom(self, df, idx):
        """
        Detect a Tweezer Bottom pattern:
        - First candle is bearish
        - Second candle is bullish
        - Both candles have similar lows
        - Appears in a downtrend
        """
        if idx < 1 or idx >= len(df):
            return False
            
        first_open = df['open'].iloc[idx-1]
        first_close = df['close'].iloc[idx-1]
        first_low = df['low'].iloc[idx-1]
        second_open = df['open'].iloc[idx]
        second_close = df['close'].iloc[idx]
        second_low = df['low'].iloc[idx]
        
        # First candle is bearish
        if not self.is_bearish_candle(first_open, first_close):
            return False
            
        # Second candle is bullish
        if not self.is_bullish_candle(second_open, second_close):
            return False
            
        # Both candles have similar lows
        low_diff = abs(first_low - second_low)
        avg_body = (self.calculate_body_size(first_open, first_close) + 
                  self.calculate_body_size(second_open, second_close)) / 2
        
        if low_diff > avg_body * 0.1:
            return False
            
        return True
    
    def detect_tweezer_top(self, df, idx):
        """
        Detect a Tweezer Top pattern:
        - First candle is bullish
        - Second candle is bearish
        - Both candles have similar highs
        - Appears in an uptrend
        """
        if idx < 1 or idx >= len(df):
            return False
            
        first_open = df['open'].iloc[idx-1]
        first_close = df['close'].iloc[idx-1]
        first_high = df['high'].iloc[idx-1]
        second_open = df['open'].iloc[idx]
        second_close = df['close'].iloc[idx]
        second_high = df['high'].iloc[idx]
        
        # First candle is bullish
        if not self.is_bullish_candle(first_open, first_close):
            return False
            
        # Second candle is bearish
        if not self.is_bearish_candle(second_open, second_close):
            return False
            
        # Both candles have similar highs
        high_diff = abs(first_high - second_high)
        avg_body = (self.calculate_body_size(first_open, first_close) + 
                  self.calculate_body_size(second_open, second_close)) / 2
        
        if high_diff > avg_body * 0.1:
            return False
            
        return True