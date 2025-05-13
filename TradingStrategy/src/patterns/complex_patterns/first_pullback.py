"""
First Pullback Pattern

The first pullback after a stock meets all 5 conditions.
This is Ross Cameron's most reliable entry signal.
"""

from ..base_pattern import BasePattern
import numpy as np


class FirstPullbackPattern(BasePattern):
    """
    Detects the First Pullback pattern - the initial pullback after 
    a stock starts trending up and meets all trading conditions.
    
    This pattern consists of:
    1. Initial surge/momentum (stock meeting 5 conditions)
    2. First pullback (brief price retracement)
    3. Resumption of uptrend (candle over candle)
    """
    
    def __init__(self):
        super().__init__(
            name="First Pullback",
            pattern_type="complex",
            min_candles_required=10
        )
        
        # Pattern parameters
        self.min_surge_gain_pct = 2.0    # Minimum gain for initial surge
        self.max_pullback_pct = 50       # Maximum pullback from surge high
        self.min_pullback_candles = 1    # Minimum candles in pullback
        self.max_pullback_candles = 5    # Maximum candles in pullback
    
    def detect(self, candles):
        """
        Detect First Pullback pattern in candlestick data.
        
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
        
        # Need enough candles
        if len(candles) < self.min_candles_required:
            return None
        
        # Look for the pattern in recent candles
        lookback = min(20, len(candles))
        recent_candles = candles.iloc[-lookback:]
        
        # Find initial surge
        surge_result = self._find_initial_surge(recent_candles)
        
        if not surge_result:
            return None
        
        # Find pullback after surge
        pullback_result = self._find_pullback(recent_candles, surge_result)
        
        if not pullback_result:
            return None
        
        # Check for entry signal (candle over candle)
        entry_result = self._check_entry_signal(recent_candles, pullback_result)
        
        if not entry_result:
            return None
        
        # Create complete pattern result
        return self._create_pattern_result(
            recent_candles, surge_result, pullback_result, entry_result
        )
    
    def _find_initial_surge(self, candles):
        """
        Find the initial surge that indicates momentum.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Recent candlestick data
            
        Returns:
        --------
        dict or None: Surge information if found
        """
        # Look for strong upward movement in first part of data
        for start_idx in range(len(candles) - 8):
            for end_idx in range(start_idx + 2, min(start_idx + 7, len(candles) - 3)):
                segment = candles.iloc[start_idx:end_idx + 1]
                
                # Calculate the surge
                start_price = segment.iloc[0]['low']
                end_price = segment.iloc[-1]['high']
                gain_pct = ((end_price - start_price) / start_price) * 100
                
                # Check if this qualifies as a surge
                if gain_pct >= self.min_surge_gain_pct:
                    # Verify strong momentum (mostly green candles)
                    green_candles = sum(1 for _, candle in segment.iterrows() 
                                       if self.is_bullish_candle(candle))
                    
                    if green_candles >= len(segment) * 0.6:  # 60% green minimum
                        return {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'start_price': start_price,
                            'end_price': end_price,
                            'high_price': segment['high'].max(),
                            'gain_pct': gain_pct
                        }
        
        return None
    
    def _find_pullback(self, candles, surge_result):
        """
        Find the pullback after the initial surge.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        surge_result: dict
            Information about the initial surge
            
        Returns:
        --------
        dict or None: Pullback information if found
        """
        surge_end_idx = surge_result['end_idx']
        surge_high = surge_result['high_price']
        
        # Look for pullback starting after surge
        pullback_start_idx = surge_end_idx + 1
        
        # Don't look too far ahead
        max_search_idx = min(pullback_start_idx + self.max_pullback_candles + 2, 
                            len(candles) - 1)
        
        for pullback_end_idx in range(pullback_start_idx, max_search_idx):
            segment = candles.iloc[pullback_start_idx:pullback_end_idx + 1]
            
            # Check if this is a valid pullback
            if len(segment) >= self.min_pullback_candles:
                pullback_low = segment['low'].min()
                
                # Calculate pullback percentage from surge high
                pullback_pct = ((surge_high - pullback_low) / surge_high) * 100
                
                # Pullback should not be too deep
                if 0 < pullback_pct <= self.max_pullback_pct:
                    # Should have at least one red candle
                    red_candles = sum(1 for _, candle in segment.iterrows() 
                                     if self.is_bearish_candle(candle))
                    
                    if red_candles > 0:
                        return {
                            'start_idx': pullback_start_idx,
                            'end_idx': pullback_end_idx,
                            'low_price': pullback_low,
                            'pullback_pct': pullback_pct,
                            'duration': len(segment)
                        }
        
        return None
    
    def _check_entry_signal(self, candles, pullback_result):
        """
        Check for entry signal (candle over candle) after pullback.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pullback_result: dict
            Information about the pullback
            
        Returns:
        --------
        dict or None: Entry signal information if found
        """
        pullback_end_idx = pullback_result['end_idx']
        
        # Check if we have candles after pullback
        if pullback_end_idx >= len(candles) - 1:
            return None
        
        # Get the pullback candle and the next candle
        pullback_candle = candles.iloc[pullback_end_idx]
        entry_candle = candles.iloc[pullback_end_idx + 1]
        
        # Entry criteria:
        # 1. Entry candle makes a new high (candle over candle)
        # 2. Entry candle is bullish
        # 3. MACD is positive (if available)
        # 4. Price is above 9 EMA (if available)
        
        if (entry_candle['high'] > pullback_candle['high'] and
            self.is_bullish_candle(entry_candle)):
            
            # Check technical indicators
            indicators_valid = self._validate_indicators(entry_candle)
            
            # Check volume increase (if available)
            volume_valid = self._check_volume_increase(candles, pullback_end_idx)
            
            if indicators_valid and volume_valid:
                return {
                    'idx': pullback_end_idx + 1,
                    'candle': entry_candle,
                    'trigger_high': pullback_candle['high']
                }
        
        return None
    
    def _create_pattern_result(self, candles, surge_result, pullback_result, entry_result):
        """
        Create the complete pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        surge_result: dict
            Surge information
        pullback_result: dict
            Pullback information
        entry_result: dict
            Entry signal information
            
        Returns:
        --------
        dict: Complete pattern result
        """
        entry_candle = entry_result['candle']
        
        # Entry price is above the entry candle high
        entry_price = entry_candle['high'] * 1.001
        
        # Stop loss is below the pullback low
        stop_price = pullback_result['low_price'] * 0.999
        
        # Target is based on risk-reward ratio (2:1)
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2.0)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            surge_result, pullback_result, entry_result, candles
        )
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': entry_result['idx'],
            'pattern_data': {
                'surge_gain_pct': surge_result['gain_pct'],
                'pullback_pct': pullback_result['pullback_pct'],
                'pullback_duration': pullback_result['duration']
            },
            'notes': f"First pullback after {surge_result['gain_pct']:.1f}% surge, "
                    f"{pullback_result['pullback_pct']:.1f}% retracement"
        }
        
        self.log_detection(result)
        return result
    
    def _validate_indicators(self, candle):
        """Check if technical indicators support entry."""
        # MACD should be positive
        if 'macd_line' in candle and candle['macd_line'] <= 0:
            return False
        
        # Price should be above 9 EMA
        if 'ema9' in candle and candle['close'] < candle['ema9']:
            return False
        
        # Price should be above 20 EMA
        if 'ema20' in candle and candle['close'] < candle['ema20']:
            return False
        
        # Price should be above VWAP
        if 'vwap' in candle and candle['close'] < candle['vwap']:
            return False
        
        return True
    
    def _check_volume_increase(self, candles, pullback_end_idx):
        """Check if volume increases on the entry candle."""
        if 'volume' not in candles.columns:
            return True  # Assume valid if no volume data
        
        # Compare entry candle volume to pullback average
        pullback_segment = candles.iloc[pullback_end_idx-2:pullback_end_idx+1]
        pullback_avg_volume = pullback_segment['volume'].mean()
        
        entry_candle = candles.iloc[pullback_end_idx + 1]
        entry_volume = entry_candle['volume']
        
        return entry_volume > pullback_avg_volume * 1.2  # 20% increase
    
    def _calculate_pattern_confidence(self, surge_result, pullback_result, 
                                    entry_result, candles):
        """Calculate confidence score for the pattern."""
        base_confidence = 80  # First pullback is high confidence pattern
        
        # Stronger surge = higher confidence
        if surge_result['gain_pct'] > 5:
            base_confidence += 5
        
        # Shallow pullback = higher confidence
        if pullback_result['pullback_pct'] < 30:
            base_confidence += 5
        
        # Quick pullback = higher confidence
        if pullback_result['duration'] <= 3:
            base_confidence += 5
        
        # Check overall trend
        trend = self.calculate_trend(candles)
        if trend == 'uptrend':
            base_confidence += 5
        
        return min(100, base_confidence)