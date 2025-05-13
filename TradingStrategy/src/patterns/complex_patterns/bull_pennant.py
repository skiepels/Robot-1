"""
Bull Pennant Pattern

A continuation pattern similar to the Bull Flag but with converging trendlines
forming a symmetrical triangle during the consolidation phase.
"""

from ..base_pattern import BasePattern
import numpy as np


class BullPennantPattern(BasePattern):
    """
    Detects the Bull Pennant pattern for momentum trading.
    
    The pattern consists of:
    1. A strong upward move (the pole)
    2. A triangular consolidation with converging trendlines (the pennant)
    3. A breakout from the pennant formation
    """
    
    def __init__(self):
        super().__init__(
            name="Bull Pennant",
            pattern_type="complex",
            min_candles_required=12  # Need enough for pole and pennant
        )
        
        # Pattern parameters
        self.min_pole_gain_pct = 5.0      # Minimum 5% move for the pole
        self.min_pennant_candles = 4      # Minimum candles for pennant
        self.max_pennant_candles = 10     # Maximum candles for pennant
        self.convergence_threshold = 0.8   # How much trendlines should converge
    
    def detect(self, candles):
        """
        Detect Bull Pennant pattern in candlestick data.
        
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
        
        # Look for potential pole formation
        for pole_start in range(len(candles) - self.min_candles_required, -1, -1):
            # Try to identify a pole starting from this point
            pole_result = self._identify_pole(candles, pole_start)
            
            if pole_result:
                pole_end = pole_result['end_idx']
                
                # Look for pennant formation after the pole
                pennant_result = self._identify_pennant(candles, pole_end)
                
                if pennant_result:
                    # Check for breakout from pennant
                    breakout_result = self._check_breakout(candles, pennant_result)
                    
                    if breakout_result:
                        # We have a complete bull pennant pattern
                        return self._create_pattern_result(
                            candles, pole_result, pennant_result, breakout_result
                        )
        
        return None
    
    def _identify_pole(self, candles, start_idx):
        """
        Identify the pole (strong upward move) of the bull pennant.
        
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
        for end_idx in range(start_idx + 2, min(start_idx + 8, len(candles))):
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
    
    def _identify_pennant(self, candles, pole_end_idx):
        """
        Identify the pennant (triangular consolidation) after the pole.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pole_end_idx: int
            End index of the pole
            
        Returns:
        --------
        dict or None: Pennant information if found
        """
        pennant_start_idx = pole_end_idx + 1
        
        # Look for pennant formation
        for pennant_end_idx in range(
            pennant_start_idx + self.min_pennant_candles,
            min(pennant_start_idx + self.max_pennant_candles, len(candles) - 1)
        ):
            segment = candles.iloc[pennant_start_idx:pennant_end_idx + 1]
            
            # Check if this segment forms a valid pennant
            pennant_info = self._analyze_pennant_formation(segment)
            
            if pennant_info and pennant_info['is_valid']:
                pennant_info.update({
                    'start_idx': pennant_start_idx,
                    'end_idx': pennant_end_idx,
                    'duration': len(segment)
                })
                return pennant_info
        
        return None
    
    def _analyze_pennant_formation(self, segment):
        """
        Analyze if a segment forms a valid pennant pattern.
        
        Parameters:
        -----------
        segment: pandas.DataFrame
            Potential pennant segment
            
        Returns:
        --------
        dict or None: Pennant analysis results
        """
        if len(segment) < self.min_pennant_candles:
            return None
        
        highs = segment['high'].values
        lows = segment['low'].values
        indices = np.arange(len(segment))
        
        try:
            # Fit trendlines to highs and lows
            upper_slope, upper_intercept = np.polyfit(indices, highs, 1)
            lower_slope, lower_intercept = np.polyfit(indices, lows, 1)
            
            # For a pennant, upper trendline should slope down and lower should slope up
            if upper_slope >= 0 or lower_slope <= 0:
                return None
            
            # Calculate convergence
            start_width = (upper_intercept - lower_intercept)
            end_width = ((upper_slope + upper_intercept) * len(segment) - 
                        (lower_slope + lower_intercept) * len(segment))
            
            if start_width <= 0:
                return None
            
            convergence_ratio = abs(end_width) / start_width
            
            # Check if trendlines are converging enough
            if convergence_ratio > self.convergence_threshold:
                return None
            
            # Calculate apex (where trendlines meet)
            if upper_slope != lower_slope:
                apex_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
            else:
                apex_x = float('inf')
            
            # Apex should be ahead (pennant should be pointing forward)
            if apex_x <= len(segment):
                return None
            
            return {
                'is_valid': True,
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'convergence_ratio': convergence_ratio,
                'apex_distance': apex_x - len(segment),
                'resistance_level': upper_slope * len(segment) + upper_intercept
            }
            
        except np.linalg.LinAlgError:
            # Could not fit trendlines
            return None
    
    def _check_breakout(self, candles, pennant_result):
        """
        Check for breakout from the pennant pattern.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pennant_result: dict
            Pennant information
            
        Returns:
        --------
        dict or None: Breakout information if found
        """
        pennant_end_idx = pennant_result['end_idx']
        
        if pennant_end_idx >= len(candles) - 1:
            return None
        
        # Get the breakout candle
        breakout_candle = candles.iloc[pennant_end_idx + 1]
        resistance_level = pennant_result['resistance_level']
        
        # Breakout criteria:
        # 1. Close above the upper trendline
        # 2. Bullish candle
        # 3. Increased volume
        
        if (breakout_candle['close'] > resistance_level and 
            self.is_bullish_candle(breakout_candle)):
            
            # Check volume confirmation if available
            volume_confirmed = True
            if 'volume' in candles.columns:
                pennant_avg_volume = candles.iloc[
                    pennant_result['start_idx']:pennant_result['end_idx']+1
                ]['volume'].mean()
                
                volume_confirmed = breakout_candle['volume'] > pennant_avg_volume * 1.3
            
            # Check technical indicators
            indicators_confirmed = self._check_indicators(breakout_candle)
            
            if volume_confirmed and indicators_confirmed:
                return {
                    'idx': pennant_end_idx + 1,
                    'candle': breakout_candle,
                    'breakout_level': resistance_level,
                    'volume_confirmed': volume_confirmed
                }
        
        return None
    
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
    
    def _create_pattern_result(self, candles, pole_result, pennant_result, breakout_result):
        """
        Create the final pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pole_result: dict
            Pole information
        pennant_result: dict
            Pennant information
        breakout_result: dict
            Breakout information
            
        Returns:
        --------
        dict: Complete pattern result
        """
        breakout_candle = breakout_result['candle']
        
        # Entry is slightly above the breakout candle high
        entry_price = breakout_candle['high'] * 1.001
        
        # Stop loss is below the pennant low
        pennant_segment = candles.iloc[
            pennant_result['start_idx']:pennant_result['end_idx']+1
        ]
        stop_price = pennant_segment['low'].min() * 0.999
        
        # Target based on pole height (measured move)
        pole_height = pole_result['height']
        target_price = entry_price + pole_height
        
        # Calculate pattern confidence
        confidence = self._calculate_pattern_confidence(
            pole_result, pennant_result, breakout_result, candles
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
                'pennant_duration': pennant_result['duration'],
                'convergence_ratio': pennant_result['convergence_ratio'],
                'apex_distance': pennant_result['apex_distance'],
                'breakout_volume_confirmed': breakout_result['volume_confirmed']
            },
            'notes': f"Bull pennant with {pole_result['gain_pct']:.1f}% pole, "
                    f"{pennant_result['duration']} candle pennant, "
                    f"convergence ratio: {pennant_result['convergence_ratio']:.2f}"
        }
        
        self.log_detection(result)
        return result
    
    def _calculate_pattern_confidence(self, pole_result, pennant_result, 
                                    breakout_result, candles):
        """Calculate confidence score for the pattern."""
        base_confidence = 70
        
        # Stronger pole = higher confidence
        if pole_result['gain_pct'] > 10:
            base_confidence += 10
        elif pole_result['gain_pct'] > 7:
            base_confidence += 5
        
        # Good convergence = higher confidence
        if pennant_result['convergence_ratio'] < 0.3:
            base_confidence += 10
        elif pennant_result['convergence_ratio'] < 0.5:
            base_confidence += 5
        
        # Proper apex distance = higher confidence
        if 2 < pennant_result['apex_distance'] < 10:
            base_confidence += 5
        
        # Volume confirmation
        if breakout_result['volume_confirmed']:
            base_confidence += 10
        
        # Check overall trend
        trend = self.calculate_trend(candles)
        if trend == 'uptrend':
            base_confidence += 5
        
        return min(100, base_confidence)