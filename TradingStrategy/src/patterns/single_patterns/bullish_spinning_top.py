"""
Bullish Spinning Top Pattern

A single candlestick with a small body in the middle of the candle's range,
with wicks on both sides. In an uptrend, it can indicate potential continuation.
"""

from ..base_pattern import BasePattern


class BullishSpinningTopPattern(BasePattern):
    """
    Detects the Bullish Spinning Top candlestick pattern.
    
    A Bullish Spinning Top is characterized by:
    - Small bullish (green) body
    - Located in the middle of the candle's trading range
    - Upper and lower wicks of similar size (relatively balanced)
    - Appears in an uptrend or at support levels
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Spinning Top",
            pattern_type="single",
            min_candles_required=5  # Need context
        )
        
        # Pattern parameters
        self.max_body_size_ratio = 0.3   # Body should be small relative to range
        self.wick_balance_threshold = 0.6 # Wicks should be relatively balanced
    
    def detect(self, candles):
        """
        Detect Bullish Spinning Top pattern in candlestick data.
        
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
        
        # Focus on the last candle
        current_candle = candles.iloc[-1]
        
        # Must be a bullish candle (close > open)
        if not self.is_bullish_candle(current_candle):
            return None
        
        # Calculate candle characteristics
        body_size = self.calculate_body_size(current_candle)
        upper_wick = self.calculate_upper_wick(current_candle)
        lower_wick = self.calculate_lower_wick(current_candle)
        total_range = self.calculate_range(current_candle)
        
        # Avoid division by zero
        if total_range == 0:
            return None
        
        # Check Bullish Spinning Top criteria
        
        # 1. Small body relative to range
        if (body_size / total_range) > self.max_body_size_ratio:
            return None
        
        # 2. Body should be near the middle of the range
        body_low = current_candle['open']
        body_high = current_candle['close']
        
        # Calculate how centered the body is (0 = perfect center)
        center_offset = abs(((body_low + body_high) / 2 - 
                           (current_candle['high'] + current_candle['low']) / 2) / total_range)
        
        if center_offset > 0.2:  # Not centered enough
            return None
        
        # 3. Wicks should be relatively balanced
        if upper_wick == 0 or lower_wick == 0:
            return None
            
        wick_ratio = min(upper_wick, lower_wick) / max(upper_wick, lower_wick)
        
        if wick_ratio < self.wick_balance_threshold:
            return None
        
        # Determine trend for context
        trend = self.calculate_trend(candles.iloc[:-1])
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(body_size, upper_wick, lower_wick, total_range, wick_ratio)
        
        # Volume confirmation
        volume_confirmation = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=(trend == 'uptrend'),
            volume_confirmation=volume_confirmation
        )
        
        # Create pattern result
        return self._create_pattern_result(candles, current_candle, confidence, trend)
    
    def _calculate_pattern_score(self, body_size, upper_wick, lower_wick, total_range, wick_ratio):
        """Calculate pattern strength based on ideal proportions."""
        score = 50  # Base score
        
        # 1. Small body (smaller is better)
        body_ratio = body_size / total_range
        if body_ratio < 0.1:
            score += 20  # Very small body
        elif body_ratio < 0.2:
            score += 10
        
        # 2. Balanced wicks (more balanced is better)
        if wick_ratio > 0.8:
            score += 20  # Very balanced wicks
        elif wick_ratio > 0.7:
            score += 10
        
        # 3. Body position (closer to center is better)
        # Already checked in main function
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        current_volume = candles.iloc[-1]['volume']
        avg_volume = candles.iloc[-4:-1]['volume'].mean()
        
        # Volume should be near or below average for indecision patterns
        return current_volume <= avg_volume * 1.2
    
    def _create_pattern_result(self, candles, spinning_top, confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        spinning_top: pandas.Series
            The Bullish Spinning Top candle
        confidence: float
            Pattern confidence score
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Determine direction based on trend context
        direction = "neutral"
        
        if trend == "uptrend":
            direction = "bullish"  # Continuation in uptrend
        
        # For trading parameters, we'll use a breakout approach
        # Entry above the spinning top high
        entry_price = spinning_top['high'] * 1.001
        
        # Stop below the spinning top low
        stop_price = spinning_top['low'] * 0.999
        
        # Target based on risk-reward
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2)  # 2:1 reward-risk
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'body_size': self.calculate_body_size(spinning_top),
                'upper_wick': self.calculate_upper_wick(spinning_top),
                'lower_wick': self.calculate_lower_wick(spinning_top),
                'body_to_range_ratio': self.calculate_body_size(spinning_top) / self.calculate_range(spinning_top)
            },
            'notes': f"Bullish Spinning Top at ${spinning_top['close']:.2f} with {confidence:.1f}% confidence"
        }
        
        self.log_detection(result)
        return result