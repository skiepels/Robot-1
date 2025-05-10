"""
Micro Pullback Strategy

This module implements the Micro Pullback pattern detection strategy for the dynamic backtesting system.
The Micro Pullback is a pattern where price makes a brief, shallow pullback in the context of
a strong uptrend, offering a lower-risk entry point.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MicroPullbackStrategy:
    """
    Implements detection of the Micro Pullback pattern.
    """
    
    def __init__(self, lookback=5, min_uptrend_pct=2.0):
        """
        Initialize the Micro Pullback strategy.
        
        Parameters:
        -----------
        lookback: int
            Number of candles to look back for pattern formation
        min_uptrend_pct: float
            Minimum percentage increase required for the prior uptrend
        """
        self.lookback = lookback
        self.min_uptrend_pct = min_uptrend_pct
    
    def detect_micro_pullback(self, df):
        """
        Detect the Micro Pullback pattern.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        dict or None: Pattern information if detected, None otherwise
        """
        # Need at least 5 candles to identify pattern
        if len(df) < self.lookback:
            return None
            
        # Get recent data
        recent_data = df.iloc[-self.lookback:]
        
        # Check for strong uptrend (first 3 candles)
        first_3_candles = recent_data.iloc[:3]
        start_price = first_3_candles['open'].iloc[0]
        end_price = first_3_candles['close'].iloc[-1]
        uptrend_pct = (end_price / start_price - 1) * 100
        
        if uptrend_pct < self.min_uptrend_pct:
            return None
            
        # Look for a micro pullback (1-2 candles)
        pullback_candle = recent_data.iloc[-2]
        is_pullback = pullback_candle['close'] < pullback_candle['open']
        
        # Check for the bottoming tail (if any)
        body_size = abs(pullback_candle['close'] - pullback_candle['open'])
        lower_wick = min(pullback_candle['open'], pullback_candle['close']) - pullback_candle['low']
        has_bottoming_tail = lower_wick > body_size * 0.5
        
        # Check for entry candle (candle over candle)
        last_candle = recent_data.iloc[-1]
        is_candle_over_candle = last_candle['high'] > pullback_candle['high'] and last_candle['close'] > last_candle['open']
        
        # Verify with MACD if available
        macd_positive = True
        if 'macd_line' in last_candle:
            macd_positive = last_candle['macd_line'] > 0
            
        # Check if pullback stayed above 9 EMA if available
        above_ema = True
        if 'ema9' in pullback_candle:
            above_ema = pullback_candle['low'] > pullback_candle['ema9']
            
        # Check for increasing volume on the breakout if available
        volume_increasing = True
        if 'volume' in last_candle and 'volume' in pullback_candle:
            volume_increasing = last_candle['volume'] > pullback_candle['volume']
        
        # Check if all criteria are met
        if (is_pullback or has_bottoming_tail) and is_candle_over_candle and macd_positive and above_ema and volume_increasing:
            # Calculate entry, stop prices
            entry_price = last_candle['high']  # Entry above the current candle
            stop_price = pullback_candle['low']  # Stop below the pullback candle
            
            # Return pattern information
            return {
                'pattern': 'micro_pullback',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'is_valid': True,
                'uptrend_pct': uptrend_pct,
                'has_bottoming_tail': has_bottoming_tail
            }
        
        return None