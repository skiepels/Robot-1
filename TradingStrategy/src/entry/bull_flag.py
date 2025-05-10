"""
Bull Flag Strategy

This module implements the Bull Flag pattern detection strategy for the dynamic backtesting system.
The Bull Flag is a continuation pattern consisting of a strong upward move (the pole),
followed by a consolidation phase (the flag).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BullFlagStrategy:
    """
    Implements detection of the Bull Flag pattern.
    """
    
    def __init__(self, lookback=10, min_pole_length_pct=5.0, max_flag_duration=7):
        """
        Initialize the Bull Flag strategy.
        
        Parameters:
        -----------
        lookback: int
            Number of candles to look back for pattern formation
        min_pole_length_pct: float
            Minimum percentage increase required for the pole
        max_flag_duration: int
            Maximum number of candles allowed for the flag/consolidation
        """
        self.lookback = lookback
        self.min_pole_length_pct = min_pole_length_pct
        self.max_flag_duration = max_flag_duration
    
    def detect_bull_flag(self, df):
        """
        Detect the Bull Flag pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        dict or None: Pattern information if detected, None otherwise
        """
        if len(df) < self.lookback:
            return None
        
        # Get the relevant candles
        recent_data = df.iloc[-self.lookback:]
        
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
            if pole_move_pct < self.min_pole_length_pct:
                continue
            
            # Now check for consolidation/flag after the pole
            flag_start_idx = pole_end_idx
            flag_end_idx = len(recent_data) - 1
            
            # Ensure flag isn't too long
            if flag_end_idx - flag_start_idx > self.max_flag_duration:
                flag_end_idx = flag_start_idx + self.max_flag_duration
            
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
            
            # Calculate entry, stop, and target
            entry_price = final_candle['high']  # Entry above the current candle
            stop_price = flag_low  # Stop below the flag low
            
            # Return pattern information
            return {
                'pattern': 'bull_flag',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'is_valid': True,
                'pole_gain_pct': pole_move_pct,
                'flag_range_pct': flag_range_pct
            }
        
        return None