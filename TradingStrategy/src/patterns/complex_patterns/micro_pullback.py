"""
Micro Pullback Pattern

A quick, shallow retracement in an established uptrend.
Features a brief 1-3 candle pullback with bottoming tails.
"""

from ..base_pattern import BasePattern
import numpy as np


class MicroPullbackPattern(BasePattern):
    """
    Detects the Micro Pullback pattern for momentum trading.
    
    This pattern consists of:
    1. Strong established uptrend
    2. Brief shallow pullback (1-3 candles)
    3. Bottoming tail candle (optional but preferred)
    4. Candle over candle entry signal
    """
    
    def __init__(self):
        super().__init__(
            name="Micro Pullback",
            pattern_type="complex",
            min_candles_required=8
        )
        
        # Pattern parameters
        self.min_trend_candles = 5        # Minimum candles to establish trend
        self.max_pullback_candles = 3     # Maximum candles in pullback
        self.max_pullback_depth_pct = 5   # Maximum pullback from recent high
        self.min_wick_ratio = 0.5         # For bottoming tail validation
    
    def detect(self, candles):
        """
        Detect Micro Pullback pattern in candlestick data.
        
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
        
        if len(candles) < self.min_candles_required:
            return None
        
        # Focus on recent candles
        lookback = min(15, len(candles))
        recent_candles = candles.iloc[-lookback:]
        
        # Check if we're in an uptrend
        if not self._verify_uptrend(recent_candles):
            return None
        
        # Look for micro pullback
        pullback_result = self._find_micro_pullback(recent_candles)
        
        if not pullback_result:
            return None
        
        # Check for entry signal
        entry_result = self._check_entry_signal(recent_candles, pullback_result)
        
        if not entry_result:
            return None
        
        # Create pattern result
        return self._create_pattern_result(recent_candles, pullback_result, entry_result)
    
    def _verify_uptrend(self, candles):
        """
        Verify that we're in an established uptrend.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Recent candlestick data
            
        Returns:
        --------
        bool: True if in valid uptrend
        """
        # Need enough candles to establish trend
        if len(candles) < self.min_trend_candles:
            return False
        
        # Calculate trend
        trend = self.calculate_trend(candles, lookback=10)
        if trend != 'uptrend':
            return False
        
        # Check price position relative to EMAs
        last_candle = candles.iloc[-1]
        
        # Price should be above key EMAs
        if 'ema9' in last_candle and last_candle['close'] < last_candle['ema9']:
            return False
        
        if 'ema20' in last_candle and last_candle['close'] < last_candle['ema20']:
            return False
        
        # Check if we've had recent upward momentum
        recent_gain = self._calculate_recent_gain(candles)
        if recent_gain < 1.0:  # Need at least 1% recent gain
            return False
        
        return True
    
    def _find_micro_pullback(self, candles):
        """
        Find a micro pullback in the recent candles.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
            
        Returns:
        --------
        dict or None: Pullback information if found
        """
        # Look for pullback in the last few candles
        for i in range(len(candles) - 1, self.min_trend_candles, -1):
            # Check if we can form a valid pullback ending at this candle
            for pullback_length in range(1, min(self.max_pullback_candles + 1, i)):
                pullback_start_idx = i - pullback_length
                pullback_end_idx = i
                
                # Get pullback segment
                pullback_segment = candles.iloc[pullback_start_idx:pullback_end_idx + 1]
                
                # Validate this pullback
                if self._is_valid_micro_pullback(pullback_segment, candles):
                    # Get the pullback low
                    pullback_low = pullback_segment['low'].min()
                    
                    # Check for bottoming tail
                    has_bottoming_tail = self._check_bottoming_tail(pullback_segment)
                    
                    return {
                        'start_idx': pullback_start_idx,
                        'end_idx': pullback_end_idx,
                        'low_price': pullback_low,
                        'has_bottoming_tail': has_bottoming_tail,
                        'duration': len(pullback_segment)
                    }
        
        return None
    
    def _is_valid_micro_pullback(self, segment, full_candles):
        """
        Check if a segment represents a valid micro pullback.
        
        Parameters:
        -----------
        segment: pandas.DataFrame
            Potential pullback segment
        full_candles: pandas.DataFrame
            Full candlestick data for context
            
        Returns:
        --------
        bool: True if valid micro pullback
        """
        # Need at least one red candle
        red_candles = sum(1 for _, candle in segment.iterrows() 
                         if self.is_bearish_candle(candle))
        
        if red_candles == 0:
            return False
        
        # Get recent high before pullback
        pre_pullback_data = full_candles.iloc[:segment.index[0]]
        if len(pre_pullback_data) < 3:
            return False
        
        recent_high = pre_pullback_data.iloc[-3:]['high'].max()
        pullback_low = segment['low'].min()
        
        # Calculate pullback depth
        pullback_depth_pct = ((recent_high - pullback_low) / recent_high) * 100
        
        # Pullback should be shallow
        if pullback_depth_pct > self.max_pullback_depth_pct:
            return False
        
        # Pullback should stay above 9 EMA (if available)
        if 'ema9' in segment.columns:
            lowest_close = segment['close'].min()
            ema9_value = segment['ema9'].iloc[-1]
            if lowest_close < ema9_value:
                return False
        
        return True
    
    def _check_bottoming_tail(self, segment):
        """
        Check if the pullback has a bottoming tail candle.
        
        Parameters:
        -----------
        segment: pandas.DataFrame
            Pullback segment
            
        Returns:
        --------
        bool: True if bottoming tail present
        """
        # Check the last candle of pullback for bottoming tail
        last_candle = segment.iloc[-1]
        
        body_size = abs(last_candle['close'] - last_candle['open'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        total_range = last_candle['high'] - last_candle['low']
        
        if total_range == 0:
            return False
        
        # Lower wick should be significant compared to body
        if body_size == 0:
            return lower_wick > 0
        
        wick_to_body_ratio = lower_wick / body_size
        
        # Also check if lower wick is significant part of total range
        wick_to_range_ratio = lower_wick / total_range
        
        return (wick_to_body_ratio >= self.min_wick_ratio and 
                wick_to_range_ratio >= 0.3)
    
    def _check_entry_signal(self, candles, pullback_result):
        """
        Check for candle over candle entry signal.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pullback_result: dict
            Pullback information
            
        Returns:
        --------
        dict or None: Entry signal information
        """
        pullback_end_idx = pullback_result['end_idx']
        
        # Check if we have a candle after pullback
        if pullback_end_idx >= len(candles) - 1:
            return None
        
        pullback_candle = candles.iloc[pullback_end_idx]
        entry_candle = candles.iloc[pullback_end_idx + 1]
        
        # Entry criteria:
        # 1. Entry candle makes new high (candle over candle)
        # 2. Entry candle is bullish
        # 3. MACD is positive
        # 4. Price is above EMAs and VWAP
        
        if (entry_candle['high'] > pullback_candle['high'] and
            self.is_bullish_candle(entry_candle)):
            
            # Validate indicators
            if not self._validate_entry_indicators(entry_candle):
                return None
            
            # Check volume increase
            if not self._check_volume_pattern(candles, pullback_end_idx):
                return None
            
            return {
                'idx': pullback_end_idx + 1,
                'candle': entry_candle,
                'trigger_high': pullback_candle['high']
            }
        
        return None
    
    def _create_pattern_result(self, candles, pullback_result, entry_result):
        """
        Create the complete pattern result.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pullback_result: dict
            Pullback information
        entry_result: dict
            Entry signal information
            
        Returns:
        --------
        dict: Complete pattern result
        """
        entry_candle = entry_result['candle']
        
        # Entry above the high
        entry_price = entry_candle['high'] * 1.001
        
        # Stop below the pullback low
        stop_price = pullback_result['low_price'] * 0.999
        
        # Target based on 2:1 reward-risk
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2.0)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            pullback_result, entry_result, candles
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
                'pullback_duration': pullback_result['duration'],
                'has_bottoming_tail': pullback_result['has_bottoming_tail']
            },
            'notes': f"Micro pullback ({pullback_result['duration']} candles) "
                    f"{'with' if pullback_result['has_bottoming_tail'] else 'without'} "
                    f"bottoming tail"
        }
        
        self.log_detection(result)
        return result
    
    def _calculate_recent_gain(self, candles):
        """Calculate recent price gain percentage."""
        if len(candles) < 5:
            return 0
        
        recent_low = candles.iloc[-5:]['low'].min()
        current_high = candles.iloc[-1]['high']
        
        if recent_low == 0:
            return 0
        
        return ((current_high - recent_low) / recent_low) * 100
    
    def _validate_entry_indicators(self, candle):
        """Validate technical indicators for entry."""
        # MACD must be positive
        if 'macd_line' in candle and candle['macd_line'] <= 0:
            return False
        
        # Price must be above 9 EMA
        if 'ema9' in candle and candle['close'] < candle['ema9']:
            return False
        
        # Price must be above 20 EMA
        if 'ema20' in candle and candle['close'] < candle['ema20']:
            return False
        
        # Price must be above VWAP
        if 'vwap' in candle and candle['close'] < candle['vwap']:
            return False
        
        return True
    
    def _check_volume_pattern(self, candles, pullback_end_idx):
        """Check if volume pattern supports entry."""
        if 'volume' not in candles.columns:
            return True
        
        # Entry volume should be higher than pullback volume
        pullback_candle = candles.iloc[pullback_end_idx]
        entry_candle = candles.iloc[pullback_end_idx + 1]
        
        return entry_candle['volume'] > pullback_candle['volume']
    
    def _calculate_pattern_confidence(self, pullback_result, entry_result, candles):
        """Calculate confidence score for the pattern."""
        base_confidence = 75
        
        # Bottoming tail increases confidence
        if pullback_result['has_bottoming_tail']:
            base_confidence += 10
        
        # Shorter pullback = higher confidence
        if pullback_result['duration'] <= 2:
            base_confidence += 5
        
        # Strong uptrend increases confidence
        trend_strength = self._assess_trend_strength(candles)
        if trend_strength > 0.7:
            base_confidence += 10
        elif trend_strength > 0.5:
            base_confidence += 5
        
        return min(100, base_confidence)
    
    def _assess_trend_strength(self, candles):
        """Assess the strength of the uptrend (0-1 scale)."""
        if len(candles) < 5:
            return 0.5
        
        # Calculate various trend metrics
        recent_candles = candles.iloc[-10:]
        
        # 1. Percentage of green candles
        green_ratio = sum(1 for _, c in recent_candles.iterrows() 
                         if self.is_bullish_candle(c)) / len(recent_candles)
        
        # 2. Price position relative to EMAs
        last_candle = candles.iloc[-1]
        ema_score = 0
        
        if 'ema9' in last_candle and last_candle['close'] > last_candle['ema9']:
            ema_score += 0.33
        if 'ema20' in last_candle and last_candle['close'] > last_candle['ema20']:
            ema_score += 0.33
        if 'ema200' in last_candle and last_candle['close'] > last_candle['ema200']:
            ema_score += 0.34
        
        # 3. Recent price gain
        recent_gain = self._calculate_recent_gain(candles)
        gain_score = min(recent_gain / 10, 1.0)  # Cap at 10% gain
        
        # Weighted average
        trend_strength = (green_ratio * 0.3 + ema_score * 0.4 + gain_score * 0.3)
        
        return trend_strength