"""
Gravestone Doji Pattern

A single candlestick pattern where the open and close are the same (or very close)
and occur at or near the low, with a long upper shadow and practically no lower shadow.
It often signals a bearish reversal when found at the top of an uptrend.
"""

from ..base_pattern import BasePattern


class GravestoneDojiPattern(BasePattern):
    """
    Detects the Gravestone Doji candlestick pattern.
    
    A Gravestone Doji is characterized by:
    - Open and close prices at or very near the low
    - Very small or nonexistent body
    - Little to no lower shadow
    - Long upper shadow (at least 3x the body size)
    - Often indicates a bearish reversal at the top of an uptrend
    """
    
    def __init__(self):
        super().__init__(
            name="Gravestone Doji",
            pattern_type="single",
            min_candles_required=5  # Need context
        )
        
        # Pattern parameters
        self.body_threshold = 0.1  # Maximum body to range ratio
        self.max_lower_shadow_ratio = 0.1  # Maximum lower shadow to range ratio
        self.min_upper_shadow_ratio = 0.7  # Minimum upper shadow to range ratio
    
    def detect(self, candles):
        """
        Detect Gravestone Doji pattern in candlestick data.
        
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
        
        # Calculate candle characteristics
        body_size = self.calculate_body_size(current_candle)
        upper_shadow = self.calculate_upper_wick(current_candle)
        lower_shadow = self.calculate_lower_wick(current_candle)
        total_range = self.calculate_range(current_candle)
        
        # Avoid division by zero
        if total_range == 0:
            return None
        
        # Check Gravestone Doji criteria
        
        # 1. Small or nonexistent body
        body_ratio = body_size / total_range
        if body_ratio > self.body_threshold:
            return None
        
        # 2. Little to no lower shadow
        lower_shadow_ratio = lower_shadow / total_range
        if lower_shadow_ratio > self.max_lower_shadow_ratio:
            return None
        
        # 3. Long upper shadow
        upper_shadow_ratio = upper_shadow / total_range
        if upper_shadow_ratio < self.min_upper_shadow_ratio:
            return None
        
        # Determine trend for context
        trend = self.calculate_trend(candles.iloc[:-1])
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(body_ratio, upper_shadow_ratio, lower_shadow_ratio)
        
        # Volume confirmation
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=(trend == 'uptrend'),  # Most significant in uptrend
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(candles, current_candle, confidence, trend)
    
    def _calculate_pattern_score(self, body_ratio, upper_shadow_ratio, lower_shadow_ratio):
        """Calculate pattern strength based on ideal proportions."""
        score = 50  # Base score
        
        # 1. Tiny body (smaller is better)
        if body_ratio < 0.05:
            score += 20  # Very small body
        elif body_ratio < 0.1:
            score += 10
        
        # 2. No lower shadow (smaller is better)
        if lower_shadow_ratio < 0.05:
            score += 20  # Minimal lower shadow
        elif lower_shadow_ratio < 0.1:
            score += 10
        
        # 3. Long upper shadow (longer is better)
        if upper_shadow_ratio > 0.9:
            score += 20  # Very long upper shadow
        elif upper_shadow_ratio > 0.8:
            score += 10
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        current_volume = candles.iloc[-1]['volume']
        avg_volume = candles.iloc[-4:-1]['volume'].mean()
        
        # Higher volume increases significance of Gravestone Doji
        return current_volume > avg_volume * 1.2
    
    def _create_pattern_result(self, candles, doji_candle, confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        doji_candle: pandas.Series
            The Gravestone Doji candle
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
            direction = "bearish"  # Potential reversal in uptrend
            
        # Entry below the Gravestone Doji low
        entry_price = doji_candle['low'] * 0.999
        
        # Stop above the Gravestone Doji high
        stop_price = doji_candle['high'] * 1.001
        
        # Target based on risk-reward
        risk = stop_price - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk
        
        # Ensure target is positive
        if target_price <= 0:
            target_price = entry_price * 0.95  # Default 5% down
        
        # Notes based on context
        if trend == "uptrend":
            significance = "potential bearish reversal"
        else:
            significance = "bearish confirmation"
            
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'body_size': self.calculate_body_size(doji_candle),
                'upper_shadow': self.calculate_upper_wick(doji_candle),
                'lower_shadow': self.calculate_lower_wick(doji_candle),
                'total_range': self.calculate_range(doji_candle)
            },
            'notes': f"Gravestone Doji at ${doji_candle['close']:.2f} indicating {significance}"
        }
        
        self.log_detection(result)
        return result