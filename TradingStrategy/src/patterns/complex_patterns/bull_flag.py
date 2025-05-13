"""
Bull Flag Pattern

A continuation pattern that forms after a strong upward move (the pole),
followed by a consolidation period (the flag). Part of Ross Cameron's strategy.
"""

from ..base_pattern import BasePattern
import numpy as np


class BullFlagPattern(BasePattern):
    """
    Detects the Bull Flag pattern for momentum trading.
    
    The pattern consists of:
    1. A strong upward move (the pole)
    2. A consolidation period (the flag) 
    3. A breakout from the consolidation
    """
    
    def __init__(self):
        super().__init__(
            name="Bull Flag",
            pattern_type="complex",
            min_candles_required=15  # Need enough for pole and flag
        )
        
        # Pattern parameters
        self.min_pole_gain_pct = 5.0  # Minimum 5% move for the pole
        self.max_flag_duration = 7     # Maximum candles for flag formation
        self.max_flag_retrace_pct = 50 # Flag can't retrace more than 50% of pole
    
    def detect(self, candles):
        """
        Detect Bull Flag pattern in candlestick data.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            OHLCV candlestick data with indicators
            
        Returns:
        --------
        dict or None: Pattern detection result
        """
        # Validate candlestick data
        if not self.validate_candles(candles):
            return None
        
        # We need enough candles to form a pattern
        if len(candles) < self.min_candles_required:
            return None
        
        # Look for potential pole formation
        for pole_start in range(len(candles) - self.min_candles_required, -1, -1):
            # Try to identify a pole starting from this point
            pole_result = self._identify_pole(candles, pole_start)
            
            if pole_result:
                pole_end = pole_result['end_idx']
                
                # Look for flag formation after the pole
                flag_result = self._identify_flag(candles, pole_end)
                
                if flag_result:
                    # Check for breakout from flag
                    breakout_result = self._check_breakout(candles, flag_result['end_idx'])
                    
                    if breakout_result:
                        # We have a complete bull flag pattern
                        return self._create_pattern_result(
                            candles, pole_result, flag_result, breakout_result
                        )
        
        return None
    
    def _identify_pole(self, candles, start_idx):
        """
        Identify the pole (strong upward move) of the bull flag.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        start_idx: int
            Starting index to look for pole
            
        Returns:
        --------
        dict or None: Pole information if found
        """
        # Look for a strong upward move
        for end_idx in range(start_idx + 2, min(start_idx + 10, len(candles))):
            segment = candles.iloc[start_idx:end_idx + 1]
            
            # Calculate the move
            start_price = segment.iloc[0]['close']
            end_price = segment.iloc[-1]['close']
            gain_pct = ((end_price - start_price) / start_price) * 100
            
            # Check if this is a valid pole
            if gain_pct >= self.min_pole_gain_pct:
                # Verify it's a relatively straight move up
                if self._is_strong_uptrend(segment):
                    return {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_price': start_price,
                        'end_price': end_price,
                        'gain_pct': gain_pct,
                        'height': end_price - start_price
                    }
        
        return None
    
    def _identify_flag(self, candles, pole_end_idx):
        """
        Identify the flag (consolidation) after the pole.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pole_end_idx: int
            End index of the pole
            
        Returns:
        --------
        dict or None: Flag information if found
        """
        # Look for consolidation after the pole
        flag_start_idx = pole_end_idx + 1
        
        # Don't look beyond maximum flag duration
        max_flag_end = min(flag_start_idx + self.max_flag_duration, len(candles) - 1)
        
        for flag_end_idx in range(flag_start_idx + 1, max_flag_end):
            segment = candles.iloc[flag_start_idx:flag_end_idx + 1]
            
            # Check if this segment forms a valid flag
            if self._is_valid_flag(segment, candles.iloc[pole_end_idx]):
                return {
                    'start_idx': flag_start_idx,
                    'end_idx': flag_end_idx,
                    'high': segment['high'].max(),
                    'low': segment['low'].min(),
                    'duration': len(segment)
                }
        
        return None
    
    def _check_breakout(self, candles, flag_end_idx):
        """
        Check for breakout from the flag pattern.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        flag_end_idx: int
            End index of the flag
            
        Returns:
        --------
        dict or None: Breakout information if found
        """
        if flag_end_idx >= len(candles) - 1:
            return None
        
        # Get the flag high (resistance level)
        flag_segment = candles.iloc[flag_end_idx - 5:flag_end_idx + 1]
        flag_high = flag_segment['high'].max()
        
        # Check the next candle for breakout
        breakout_candle = candles.iloc[flag_end_idx + 1]
        
        # Breakout criteria:
        # 1. Close above flag high
        # 2. Increased volume (if available)
        # 3. Bullish candle
        
        if (breakout_candle['close'] > flag_high and 
            self.is_bullish_candle(breakout_candle)):
            
            # Check volume confirmation if available
            volume_confirmed = True
            if 'volume' in candles.columns:
                avg_flag_volume = flag_segment['volume'].mean()
                volume_confirmed = breakout_candle['volume'] > avg_flag_volume
            
            # Check technical indicators if available
            indicators_confirmed = self._check_indicators(breakout_candle)
            
            if volume_confirmed and indicators_confirmed:
                return {
                    'idx': flag_end_idx + 1,
                    'candle': breakout_candle,
                    'breakout_level': flag_high,
                    'volume_confirmed': volume_confirmed
                }
        
        return None
    
    def _create_pattern_result(self, candles, pole_result, flag_result, breakout_result):
        """
        Create the final pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pole_result: dict
            Pole information
        flag_result: dict
            Flag information
        breakout_result: dict
            Breakout information
            
        Returns:
        --------
        dict: Complete pattern result
        """
        breakout_candle = breakout_result['candle']
        
        # Entry is slightly above the breakout candle high
        entry_price = breakout_candle['high'] * 1.001
        
        # Stop loss is below the flag low
        stop_price = flag_result['low'] * 0.999
        
        # Target based on pole height (measured move)
        pole_height = pole_result['height']
        target_price = entry_price + pole_height
        
        # Calculate pattern confidence
        confidence = self._calculate_pattern_confidence(
            pole_result, flag_result, breakout_result, candles
        )
        
        # Prepare result
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': breakout_result['idx'],
            'pattern_data': {
                'pole_gain_pct': pole_result['gain_pct'],
                'flag_duration': flag_result['duration'],
                'breakout_volume_confirmed': breakout_result['volume_confirmed']
            },
            'notes': f"Bull flag with {pole_result['gain_pct']:.1f}% pole, "
                    f"{flag_result['duration']} candle flag"
        }
        
        self.log_detection(result)
        return result
    
    def _is_strong_uptrend(self, segment):
        """Check if a segment shows a strong uptrend."""
        # Most candles should be green
        green_candles = sum(1 for _, candle in segment.iterrows() 
                           if self.is_bullish_candle(candle))
        
        if green_candles < len(segment) * 0.6:  # At least 60% green
            return False
        
        # Price should trend up consistently
        lows = segment['low'].values
        highs = segment['high'].values
        
        # Check for higher lows and higher highs
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        
        return higher_lows >= len(lows) * 0.5 and higher_highs >= len(highs) * 0.5
    
    def _is_valid_flag(self, segment, pole_end_candle):
        """Check if a segment forms a valid flag."""
        # Flag should be a consolidation (not trending strongly)
        high_range = segment['high'].max() - segment['high'].min()
        low_range = segment['low'].max() - segment['low'].min()
        
        avg_range = (high_range + low_range) / 2
        avg_price = segment['close'].mean()
        
        # Consolidation should be relatively tight (less than 5% range)
        if (avg_range / avg_price) > 0.05:
            return False
        
        # Flag shouldn't retrace too much of the pole
        pole_high = pole_end_candle['high']
        pole_low = pole_end_candle['low']
        flag_low = segment['low'].min()
        
        retrace_pct = ((pole_high - flag_low) / (pole_high - pole_low)) * 100
        
        if retrace_pct > self.max_flag_retrace_pct:
            return False
        
        return True
    
    def _check_indicators(self, candle):
        """Check if technical indicators support the breakout."""
        # Check MACD if available
        if 'macd_line' in candle and candle['macd_line'] <= 0:
            return False
        
        # Check if price is above key EMAs
        if 'ema9' in candle and candle['close'] < candle['ema9']:
            return False
        
        if 'ema20' in candle and candle['close'] < candle['ema20']:
            return False
        
        return True
    
    def _calculate_pattern_confidence(self, pole_result, flag_result, breakout_result, candles):
        """Calculate confidence score for the pattern."""
        base_confidence = 70
        
        # Stronger pole = higher confidence
        if pole_result['gain_pct'] > 10:
            base_confidence += 10
        elif pole_result['gain_pct'] > 7:
            base_confidence += 5
        
        # Tighter flag = higher confidence
        if flag_result['duration'] <= 5:
            base_confidence += 5
        
        # Volume confirmation
        if breakout_result['volume_confirmed']:
            base_confidence += 10
        
        # Check overall trend
        trend = self.calculate_trend(candles)
        if trend == 'uptrend':
            base_confidence += 5
        
        return min(100, base_confidence)