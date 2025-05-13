"""
Rising Three Pattern

A bullish continuation pattern consisting of a large bullish candle, 
followed by three small bearish candles that stay within the range of the first candle,
and finally another large bullish candle breaking above the first one.
"""

from ..base_pattern import BasePattern
import numpy as np


class RisingThreePattern(BasePattern):
    """
    Detects the Rising Three (Rising Three Methods) candlestick pattern.
    
    The Rising Three pattern consists of:
    - First candle: Large bullish candle
    - Next three candles: Small bearish or neutral candles contained within
      the high-low range of the first candle
    - Fifth candle: Large bullish candle that closes above the first candle's close
    - The pattern indicates bullish continuation in an uptrend
    """
    
    def __init__(self):
        super().__init__(
            name="Rising Three",
            pattern_type="triple",
            min_candles_required=12  # Need context plus 5 candles for pattern
        )
        
        # Pattern parameters
        self.large_body_threshold = 0.6  # Minimum body/range ratio for large candles
        self.small_body_threshold = 0.5  # Maximum body/range ratio for small candles
        self.containment_flexibility = 0.05  # Flexibility for containment check (5%)
    
    def detect(self, candles):
        """
        Detect Rising Three pattern in candlestick data.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            OHLCV candlestick data
            
        Returns:
        --------
        dict or None: Pattern detection result
        """
        # Validate candlestick data
        if not self.validate_candles(candles):
            return None
        
        if len(candles) < self.min_candles_required:
            return None
        
        # Get the 5 candles needed for the pattern
        first_candle = candles.iloc[-5]
        middle_candles = [candles.iloc[-4], candles.iloc[-3], candles.iloc[-2]]
        last_candle = candles.iloc[-1]
        
        # Check Rising Three criteria
        
        # 1. First candle must be a strong bullish candle
        if not self.is_bullish_candle(first_candle):
            return None
            
        first_body_ratio = self.calculate_body_size(first_candle) / self.calculate_range(first_candle)
        if first_body_ratio < self.large_body_threshold:
            return None
        
        # 2. Last candle must be a strong bullish candle
        if not self.is_bullish_candle(last_candle):
            return None
            
        last_body_ratio = self.calculate_body_size(last_candle) / self.calculate_range(last_candle)
        if last_body_ratio < self.large_body_threshold:
            return None
        
        # 3. Last candle must close above the first candle
        if last_candle['close'] <= first_candle['close']:
            return None
        
        # 4. Middle candles should be small and stay within first candle's range
        # Allow slightly exceeding the range
        first_high = first_candle['high']
        first_low = first_candle['low']
        high_limit = first_high * (1 + self.containment_flexibility)
        low_limit = first_low * (1 - self.containment_flexibility)
        
        for candle in middle_candles:
            # Middle candles should preferably be bearish, but can be neutral/small
            if self.is_bullish_candle(candle):
                body_ratio = self.calculate_body_size(candle) / self.calculate_range(candle)
                if body_ratio > self.small_body_threshold:  # Strong bullish not allowed
                    return None
            
            # Check containment
            if candle['high'] > high_limit or candle['low'] < low_limit:
                return None
        
        # 5. Middle candles should not make significant progress upward
        # Ideally they move sideways or slightly down
        middle_trend = self._check_middle_trend(middle_candles)
        if middle_trend == 'strong_uptrend':
            return None
        
        # 6. Determine if we're in an uptrend (for continuation context)
        trend = self.calculate_trend(candles.iloc[:-5])
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(
            first_candle, middle_candles, last_candle, trend
        )
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=(trend == 'uptrend'),  # Most significant in uptrend
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(
            candles, first_candle, middle_candles, last_candle, 
            confidence, trend
        )
    
    def _check_middle_trend(self, middle_candles):
        """
        Check the trend of the middle candles.
        
        Parameters:
        -----------
        middle_candles: list
            List of the three middle candles
            
        Returns:
        --------
        str: 'downtrend', 'sideways', 'uptrend', or 'strong_uptrend'
        """
        # Check the closing prices
        close_prices = [candle['close'] for candle in middle_candles]
        
        # Calculate slope
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        
        # Normalize slope by average price
        avg_price = np.mean(close_prices)
        norm_slope = slope / avg_price if avg_price > 0 else 0
        
        if norm_slope < -0.005:
            return 'downtrend'
        elif norm_slope > 0.01:
            return 'strong_uptrend'
        elif norm_slope > 0.002:
            return 'uptrend'
        else:
            return 'sideways'
    
    def _calculate_pattern_score(self, first_candle, middle_candles, last_candle, trend):
        """Calculate pattern strength based on ideal proportions."""
        score = 60  # Base score
        
        # 1. Trend context
        if trend == 'uptrend':
            score += 15  # Ideal context for continuation
        
        # 2. Strong first and last candles
        first_body_ratio = self.calculate_body_size(first_candle) / self.calculate_range(first_candle)
        last_body_ratio = self.calculate_body_size(last_candle) / self.calculate_range(last_candle)
        
        if first_body_ratio > 0.8 and last_body_ratio > 0.8:
            score += 10  # Very strong framing candles
        
        # 3. Last candle breaks above significantly
        breakout_strength = (last_candle['close'] - first_candle['close']) / first_candle['close'] * 100
        
        if breakout_strength > 2:
            score += 10  # Strong breakout
        elif breakout_strength > 1:
            score += 5
        
        # 4. Middle candles trend
        middle_trend = self._check_middle_trend(middle_candles)
        if middle_trend == 'downtrend':
            score += 10  # Ideal middle trend for pattern
        elif middle_trend == 'sideways':
            score += 5
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 5:
            return False
        
        first_candle_vol = candles.iloc[-5]['volume']
        middle_vols = [candles.iloc[-4]['volume'], 
                     candles.iloc[-3]['volume'], 
                     candles.iloc[-2]['volume']]
        last_candle_vol = candles.iloc[-1]['volume']
        
        # Ideal volume pattern: High volume on first and last candles
        # Middle candles should have lower volume
        avg_middle_vol = sum(middle_vols) / len(middle_vols)
        
        return (first_candle_vol > avg_middle_vol * 1.2 and 
               last_candle_vol > avg_middle_vol * 1.2)
    
    def _create_pattern_result(self, candles, first_candle, middle_candles, last_candle, 
                            confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First candle in pattern
        middle_candles: list
            List of the three middle candles
        last_candle: pandas.Series
            Last candle in pattern
        confidence: float
            Pattern confidence score
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Entry above the high of the last candle
        entry_price = last_candle['high'] * 1.001
        
        # Stop below the low of the pattern
        pattern_low = min(
            first_candle['low'],
            middle_candles[0]['low'],
            middle_candles[1]['low'],
            middle_candles[2]['low'],
            last_candle['low']
        ) * 0.999
        
        # Use the middle candles low to tighten stop if possible
        middle_low = min(m_candle['low'] for m_candle in middle_candles)
        stop_price = max(pattern_low, middle_low * 0.995)
        
        # Target based on risk-reward
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2)  # 2:1 reward-risk
        
        # Pattern strength description
        if confidence >= 80:
            strength = "strong"
        elif confidence >= 70:
            strength = "moderate"
        else:
            strength = "potential"
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'first_candle_close': first_candle['close'],
                'last_candle_close': last_candle['close'],
                'breakout_percent': (last_candle['close'] - first_candle['close']) / first_candle['close'] * 100,
                'middle_trend': self._check_middle_trend(middle_candles)
            },
            'notes': f"Rising Three forming a {strength} bullish continuation signal"
        }
        
        self.log_detection(result)
        return result