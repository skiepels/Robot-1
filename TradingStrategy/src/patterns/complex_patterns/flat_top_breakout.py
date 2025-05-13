"""
Flat Top Breakout Pattern

A breakout pattern where price consolidates with a flat resistance level
before breaking through on increased volume.
"""

from ..base_pattern import BasePattern
import numpy as np


class FlatTopBreakoutPattern(BasePattern):
    """
    Detects the Flat Top Breakout pattern for momentum trading.
    
    This pattern consists of:
    1. Established uptrend
    2. Consolidation with flat top (resistance level)
    3. Multiple touches of resistance
    4. Breakout above resistance on volume
    """
    
    def __init__(self):
        super().__init__(
            name="Flat Top Breakout",
            pattern_type="complex",
            min_candles_required=12
        )
        
        # Pattern parameters
        self.min_touches = 2              # Minimum touches of resistance
        self.max_resistance_deviation = 0.3  # Max % deviation for resistance level
        self.min_consolidation_candles = 5   # Minimum candles in consolidation
        self.min_breakout_volume_ratio = 1.5 # Breakout volume vs consolidation avg
    
    def detect(self, candles):
        """
        Detect Flat Top Breakout pattern in candlestick data.
        
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
        
        # Look for pattern in recent data
        lookback = min(30, len(candles))
        recent_candles = candles.iloc[-lookback:]
        
        # Find potential flat top (resistance level)
        resistance_result = self._find_flat_top_resistance(recent_candles)
        
        if not resistance_result:
            return None
        
        # Verify consolidation pattern
        consolidation_result = self._verify_consolidation(
            recent_candles, resistance_result
        )
        
        if not consolidation_result:
            return None
        
        # Check for breakout
        breakout_result = self._check_breakout(
            recent_candles, resistance_result, consolidation_result
        )
        
        if not breakout_result:
            return None
        
        # Create pattern result
        return self._create_pattern_result(
            recent_candles, resistance_result, consolidation_result, breakout_result
        )
    
    def _find_flat_top_resistance(self, candles):
        """
        Find the flat top resistance level.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Recent candlestick data
            
        Returns:
        --------
        dict or None: Resistance level information
        """
        # Look for resistance level in different windows
        for window_size in range(20, 5, -1):
            if window_size > len(candles):
                continue
            
            for start_idx in range(len(candles) - window_size):
                end_idx = start_idx + window_size
                window = candles.iloc[start_idx:end_idx]
                
                # Find potential resistance level
                resistance_info = self._identify_resistance_level(window)
                
                if resistance_info and resistance_info['touches'] >= self.min_touches:
                    resistance_info['start_idx'] = start_idx
                    resistance_info['end_idx'] = end_idx - 1
                    return resistance_info
        
        return None
    
    def _identify_resistance_level(self, window):
        """
        Identify resistance level in a window of candles.
        
        Parameters:
        -----------
        window: pandas.DataFrame
            Window of candlestick data
            
        Returns:
        --------
        dict or None: Resistance level info
        """
        # Find the highest highs
        highs = window['high'].values
        
        # Use the maximum high as initial resistance candidate
        resistance_price = np.max(highs)
        
        # Count how many times price touched this level
        touches = 0
        touch_indices = []
        
        for i, high in enumerate(highs):
            # Check if this high is close to resistance
            deviation_pct = abs(high - resistance_price) / resistance_price * 100
            
            if deviation_pct <= self.max_resistance_deviation:
                touches += 1
                touch_indices.append(i)
        
        if touches >= self.min_touches:
            # Verify these touches are spread out (not all consecutive)
            if self._touches_are_valid(touch_indices):
                return {
                    'resistance_price': resistance_price,
                    'touches': touches,
                    'touch_indices': touch_indices
                }
        
        return None
    
    def _touches_are_valid(self, touch_indices):
        """Check if resistance touches are properly distributed."""
        if len(touch_indices) < 2:
            return False
        
        # At least one touch should be separated from others
        for i in range(len(touch_indices) - 1):
            if touch_indices[i+1] - touch_indices[i] > 1:
                return True
        
        return False
    
    def _verify_consolidation(self, candles, resistance_result):
        """
        Verify proper consolidation below resistance.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        resistance_result: dict
            Resistance level information
            
        Returns:
        --------
        dict or None: Consolidation information
        """
        resistance_price = resistance_result['resistance_price']
        start_idx = resistance_result['start_idx']
        end_idx = resistance_result['end_idx']
        
        consolidation_window = candles.iloc[start_idx:end_idx+1]
        
        # Check consolidation characteristics
        if len(consolidation_window) < self.min_consolidation_candles:
            return None
        
        # Calculate consolidation metrics
        consolidation_high = consolidation_window['high'].max()
        consolidation_low = consolidation_window['low'].min()
        consolidation_range = consolidation_high - consolidation_low
        avg_price = consolidation_window['close'].mean()
        
        # Consolidation should be relatively tight
        range_pct = (consolidation_range / avg_price) * 100
        
        if range_pct > 10:  # Too wide consolidation
            return None
        
        # Most of the consolidation should be below resistance
        closes_above_resistance = sum(1 for close in consolidation_window['close'] 
                                    if close > resistance_price)
        
        if closes_above_resistance > len(consolidation_window) * 0.2:
            return None
        
        # Calculate average volume during consolidation
        avg_volume = consolidation_window['volume'].mean() if 'volume' in consolidation_window else 0
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'high': consolidation_high,
            'low': consolidation_low,
            'range_pct': range_pct,
            'avg_volume': avg_volume,
            'duration': len(consolidation_window)
        }
    
    def _check_breakout(self, candles, resistance_result, consolidation_result):
        """
        Check for breakout above resistance.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        resistance_result: dict
            Resistance information
        consolidation_result: dict
            Consolidation information
            
        Returns:
        --------
        dict or None: Breakout information
        """
        resistance_price = resistance_result['resistance_price']
        consolidation_end = consolidation_result['end_idx']
        
        # Look for breakout after consolidation
        for i in range(consolidation_end + 1, len(candles)):
            candle = candles.iloc[i]
            
            # Breakout criteria:
            # 1. Close above resistance
            # 2. Strong bullish candle
            # 3. Increased volume
            
            if candle['close'] > resistance_price:
                # Check if it's a strong bullish candle
                if not self.is_bullish_candle(candle):
                    continue
                
                # Calculate candle strength
                candle_body = abs(candle['close'] - candle['open'])
                candle_range = candle['high'] - candle['low']
                
                if candle_range == 0:
                    continue
                
                body_to_range_ratio = candle_body / candle_range
                
                if body_to_range_ratio < 0.5:  # Weak candle
                    continue
                
                # Check volume confirmation
                if 'volume' in candles.columns:
                    breakout_volume = candle['volume']
                    avg_consolidation_volume = consolidation_result['avg_volume']
                    
                    if avg_consolidation_volume > 0:
                        volume_ratio = breakout_volume / avg_consolidation_volume
                        
                        if volume_ratio < self.min_breakout_volume_ratio:
                            continue
                
                # Validate with indicators
                if not self._validate_breakout_indicators(candle):
                    continue
                
                return {
                    'idx': i,
                    'candle': candle,
                    'breakout_strength': body_to_range_ratio,
                    'volume_ratio': volume_ratio if 'volume' in candles.columns else 1.0
                }
        
        return None
    
    def _create_pattern_result(self, candles, resistance_result, consolidation_result, 
                              breakout_result):
        """
        Create the complete pattern result.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        resistance_result: dict
            Resistance information
        consolidation_result: dict
            Consolidation information
        breakout_result: dict
            Breakout information
            
        Returns:
        --------
        dict: Complete pattern result
        """
        breakout_candle = breakout_result['candle']
        resistance_price = resistance_result['resistance_price']
        
        # Entry above the breakout candle high
        entry_price = breakout_candle['high'] * 1.001
        
        # Stop below the consolidation low or below resistance
        stop_option1 = consolidation_result['low'] * 0.999
        stop_option2 = resistance_price * 0.995
        stop_price = max(stop_option1, stop_option2)  # Use the higher stop
        
        # Target based on consolidation range
        consolidation_range = consolidation_result['high'] - consolidation_result['low']
        target_price = entry_price + consolidation_range  # Measured move
        
        # Ensure minimum 2:1 reward-risk
        risk = entry_price - stop_price
        min_target = entry_price + (risk * 2.0)
        target_price = max(target_price, min_target)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            resistance_result, consolidation_result, breakout_result
        )
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': breakout_result['idx'],
            'pattern_data': {
                'resistance_price': resistance_price,
                'resistance_touches': resistance_result['touches'],
                'consolidation_duration': consolidation_result['duration'],
                'consolidation_range_pct': consolidation_result['range_pct'],
                'breakout_strength': breakout_result['breakout_strength'],
                'volume_ratio': breakout_result['volume_ratio']
            },
            'notes': f"Flat top breakout above ${resistance_price:.2f} after "
                    f"{resistance_result['touches']} touches and "
                    f"{consolidation_result['duration']} candle consolidation"
        }
        
        self.log_detection(result)
        return result
    
    def _validate_breakout_indicators(self, candle):
        """Validate technical indicators support the breakout."""
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
    
    def _calculate_pattern_confidence(self, resistance_result, consolidation_result, 
                                    breakout_result):
        """Calculate confidence score for the pattern."""
        base_confidence = 70
        
        # More touches of resistance = higher confidence
        if resistance_result['touches'] >= 4:
            base_confidence += 10
        elif resistance_result['touches'] >= 3:
            base_confidence += 5
        
        # Tighter consolidation = higher confidence
        if consolidation_result['range_pct'] < 5:
            base_confidence += 5
        
        # Strong breakout = higher confidence
        if breakout_result['breakout_strength'] > 0.7:
            base_confidence += 10
        elif breakout_result['breakout_strength'] > 0.6:
            base_confidence += 5
        
        # Volume confirmation = higher confidence
        if breakout_result['volume_ratio'] > 2.0:
            base_confidence += 10
        elif breakout_result['volume_ratio'] > 1.5:
            base_confidence += 5
        
        return min(100, base_confidence)