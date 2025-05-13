"""
New High Breakout Pattern

The first candle to make a new high after consolidation or pullback.
A key momentum entry signal in Ross Cameron's strategy.
"""

from ..base_pattern import BasePattern
import numpy as np


class NewHighBreakoutPattern(BasePattern):
    """
    Detects the New High Breakout pattern - when price breaks above
    recent resistance to new highs, signaling momentum continuation.
    
    This pattern consists of:
    1. Established price level or consolidation
    2. Break above recent highs
    3. Volume confirmation
    4. Technical indicator alignment
    """
    
    def __init__(self):
        super().__init__(
            name="New High Breakout",
            pattern_type="complex",
            min_candles_required=10
        )
        
        # Pattern parameters
        self.lookback_period = 20         # Candles to look back for highs
        self.min_consolidation_candles = 3  # Minimum consolidation before breakout
        self.min_volume_increase = 1.3     # Minimum volume increase on breakout
        self.breakout_threshold = 0.001    # How much above high to confirm breakout
    
    def detect(self, candles):
        """
        Detect New High Breakout pattern in candlestick data.
        
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
        
        # Get current candle and recent history
        current_candle = candles.iloc[-1]
        lookback_candles = candles.iloc[-(self.lookback_period + 1):-1]
        
        # Find recent high
        recent_high = lookback_candles['high'].max()
        recent_high_idx = lookback_candles['high'].idxmax()
        recent_high_position = lookback_candles.index.get_loc(recent_high_idx)
        
        # Check if current candle breaks above recent high
        if current_candle['high'] <= recent_high * (1 + self.breakout_threshold):
            return None
        
        # Verify we had consolidation before breakout
        consolidation_result = self._verify_consolidation(candles, recent_high_position)
        
        if not consolidation_result:
            return None
        
        # Check breakout quality
        breakout_quality = self._assess_breakout_quality(
            current_candle, recent_high, candles
        )
        
        if not breakout_quality['is_valid']:
            return None
        
        # Create pattern result
        return self._create_pattern_result(
            candles, current_candle, recent_high, 
            consolidation_result, breakout_quality
        )
    
    def _verify_consolidation(self, candles, high_position):
        """
        Verify there was consolidation before the new high breakout.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        high_position: int
            Position of the recent high in the lookback window
            
        Returns:
        --------
        dict or None: Consolidation information
        """
        # Get candles after the high
        candles_after_high = candles.iloc[high_position+1:-1]
        
        if len(candles_after_high) < self.min_consolidation_candles:
            return None
        
        # Calculate consolidation metrics
        consolidation_high = candles_after_high['high'].max()
        consolidation_low = candles_after_high['low'].min()
        consolidation_range = consolidation_high - consolidation_low
        avg_price = candles_after_high['close'].mean()
        
        # Check if price consolidated (didn't make new high until now)
        if consolidation_high > candles.iloc[high_position]['high']:
            return None  # Made new high during "consolidation"
        
        # Calculate range as percentage
        range_pct = (consolidation_range / avg_price) * 100 if avg_price > 0 else 0
        
        # Calculate average volume during consolidation
        avg_volume = candles_after_high['volume'].mean() if 'volume' in candles_after_high else 0
        
        return {
            'duration': len(candles_after_high),
            'high': consolidation_high,
            'low': consolidation_low,
            'range_pct': range_pct,
            'avg_volume': avg_volume
        }
    
    def _assess_breakout_quality(self, breakout_candle, recent_high, candles):
        """
        Assess the quality of the breakout.
        
        Parameters:
        -----------
        breakout_candle: pandas.Series
            The breakout candle
        recent_high: float
            The recent high being broken
        candles: pandas.DataFrame
            Full candlestick data
            
        Returns:
        --------
        dict: Breakout quality assessment
        """
        quality = {
            'is_valid': False,
            'strength': 0,
            'volume_confirmation': False,
            'indicator_alignment': False
        }
        
        # 1. Breakout candle should be bullish
        if not self.is_bullish_candle(breakout_candle):
            return quality
        
        # 2. Strong close (upper half of range)
        candle_range = breakout_candle['high'] - breakout_candle['low']
        if candle_range > 0:
            close_position = (breakout_candle['close'] - breakout_candle['low']) / candle_range
            quality['strength'] = close_position
            
            if close_position < 0.5:
                return quality
        
        # 3. Volume confirmation
        if 'volume' in candles.columns:
            recent_avg_volume = candles['volume'].iloc[-10:-1].mean()
            volume_ratio = (breakout_candle['volume'] / recent_avg_volume 
                          if recent_avg_volume > 0 else 0)
            
            quality['volume_confirmation'] = volume_ratio >= self.min_volume_increase
            
            if not quality['volume_confirmation']:
                return quality
        else:
            quality['volume_confirmation'] = True  # Assume valid if no volume data
        
        # 4. Technical indicator alignment
        quality['indicator_alignment'] = self._check_indicator_alignment(breakout_candle)
        
        if not quality['indicator_alignment']:
            return quality
        
        # 5. Check breakout conviction (how far above the high)
        breakout_margin = (breakout_candle['high'] - recent_high) / recent_high * 100
        
        if breakout_margin < 0.1:  # Too close to old high
            return quality
        
        quality['is_valid'] = True
        return quality
    
    def _check_indicator_alignment(self, candle):
        """
        Check if technical indicators support the breakout.
        
        Parameters:
        -----------
        candle: pandas.Series
            Breakout candle data
            
        Returns:
        --------
        bool: True if indicators align
        """
        # Price should be above key EMAs
        if 'ema9' in candle and candle['close'] < candle['ema9']:
            return False
        
        if 'ema20' in candle and candle['close'] < candle['ema20']:
            return False
        
        # MACD should be positive
        if 'macd_line' in candle and candle['macd_line'] <= 0:
            return False
        
        # Price should be above VWAP
        if 'vwap' in candle and candle['close'] < candle['vwap']:
            return False
        
        return True
    
    def _create_pattern_result(self, candles, breakout_candle, recent_high, 
                              consolidation_result, breakout_quality):
        """
        Create the complete pattern result.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        breakout_candle: pandas.Series
            The breakout candle
        recent_high: float
            The high that was broken
        consolidation_result: dict
            Consolidation information
        breakout_quality: dict
            Breakout quality assessment
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Entry above the breakout high
        entry_price = breakout_candle['high'] * 1.001
        
        # Stop loss options
        stop_option1 = consolidation_result['low'] * 0.999  # Below consolidation
        stop_option2 = recent_high * 0.995  # Below old resistance
        stop_option3 = breakout_candle['low'] * 0.999  # Below breakout candle
        
        # Use the highest stop (most conservative)
        stop_price = max(stop_option1, stop_option2, stop_option3)
        
        # Target calculation
        consolidation_range = consolidation_result['high'] - consolidation_result['low']
        measured_move_target = entry_price + consolidation_range
        
        # Ensure minimum 2:1 reward-risk
        risk = entry_price - stop_price
        min_target = entry_price + (risk * 2.0)
        target_price = max(measured_move_target, min_target)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            consolidation_result, breakout_quality, candles
        )
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'broken_high': recent_high,
                'breakout_high': breakout_candle['high'],
                'consolidation_duration': consolidation_result['duration'],
                'consolidation_range_pct': consolidation_result['range_pct'],
                'breakout_strength': breakout_quality['strength'],
                'volume_confirmed': breakout_quality['volume_confirmation']
            },
            'notes': f"New high breakout above ${recent_high:.2f} after "
                    f"{consolidation_result['duration']} candle consolidation"
        }
        
        self.log_detection(result)
        return result
    
    def _calculate_pattern_confidence(self, consolidation_result, breakout_quality, candles):
        """Calculate confidence score for the pattern."""
        base_confidence = 70
        
        # Tighter consolidation = higher confidence
        if consolidation_result['range_pct'] < 3:
            base_confidence += 10
        elif consolidation_result['range_pct'] < 5:
            base_confidence += 5
        
        # Longer consolidation = higher confidence
        if consolidation_result['duration'] > 10:
            base_confidence += 5
        
        # Strong breakout = higher confidence
        if breakout_quality['strength'] > 0.8:
            base_confidence += 10
        elif breakout_quality['strength'] > 0.6:
            base_confidence += 5
        
        # Volume confirmation
        if breakout_quality['volume_confirmation']:
            base_confidence += 5
        
        # Overall trend
        trend = self.calculate_trend(candles)
        if trend == 'uptrend':
            base_confidence += 5
        
        return min(100, base_confidence)