"""
Inverted Hammer Pattern

A bullish reversal pattern that forms at the bottom of a downtrend.
Characterized by a small body at the bottom of the candle's range with a long upper wick.
"""

from ..base_pattern import BasePattern


class InvertedHammerPattern(BasePattern):
    """
    Detects the Inverted Hammer candlestick pattern.
    
    An Inverted Hammer is characterized by:
    - Small body at lower end of the price range
    - Long upper shadow (at least 2x the body size)
    - Little or no lower shadow
    - Appears after a downtrend
    """
    
    def __init__(self):
        super().__init__(
            name="Inverted Hammer",
            pattern_type="single",
            min_candles_required=5  # Need context
        )
        
        # Pattern parameters
        self.min_upper_wick_ratio = 2.0  # Upper wick should be at least 2x body
        self.max_lower_wick_ratio = 0.5  # Lower wick should be < 50% of body
        self.max_body_size_ratio = 0.3   # Body should be small relative to range
    
    def detect(self, candles):
        """
        Detect Inverted Hammer pattern in candlestick data.
        
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
        
        # Determine the trend leading up to this candle
        trend = self.calculate_trend(candles.iloc[:-1])
        
        # Inverted Hammer should appear in a downtrend
        if trend != 'downtrend':
            return None
        
        # Focus on the last candle
        current_candle = candles.iloc[-1]
        
        # Calculate candle characteristics
        body_size = self.calculate_body_size(current_candle)
        upper_wick = self.calculate_upper_wick(current_candle)
        lower_wick = self.calculate_lower_wick(current_candle)
        total_range = self.calculate_range(current_candle)
        
        # Avoid division by zero
        if body_size == 0 or total_range == 0:
            return None
        
        # Check Inverted Hammer criteria
        
        # 1. Upper wick should be at least 2x the body size
        if upper_wick < (body_size * self.min_upper_wick_ratio):
            return None
        
        # 2. Lower wick should be small
        if lower_wick > (body_size * self.max_lower_wick_ratio):
            return None
        
        # 3. Body should be in the lower third of the range
        body_position = (max(current_candle['open'], current_candle['close']) - current_candle['low']) / total_range
        if body_position > 0.33:  # Body must be in lower 1/3
            return None
        
        # 4. Body should be relatively small compared to the range
        if (body_size / total_range) > self.max_body_size_ratio:
            return None
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(body_size, upper_wick, lower_wick, total_range)
        
        # Check volume confirmation
        volume_confirmed = self._check_volume_increase(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=True,
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(candles, current_candle, confidence)
    
    def _calculate_pattern_score(self, body_size, upper_wick, lower_wick, total_range):
        """Calculate how well the candle matches ideal Inverted Hammer proportions."""
        score = 50  # Base score
        
        # Upper wick should be large
        upper_wick_ratio = upper_wick / body_size if body_size > 0 else float('inf')
        if upper_wick_ratio > 3:
            score += 20
        elif upper_wick_ratio > 2:
            score += 10
        
        # Lower wick should be small
        lower_wick_ratio = lower_wick / body_size if body_size > 0 else float('inf')
        if lower_wick_ratio < 0.1:
            score += 15
        elif lower_wick_ratio < 0.3:
            score += 10
        
        # Body should be small compared to range
        body_ratio = body_size / total_range
        if body_ratio < 0.1:
            score += 15
        elif body_ratio < 0.2:
            score += 10
        
        return min(score, 100)
    
    def _check_volume_increase(self, candles):
        """Check if volume increased on the Inverted Hammer candle."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        # Compare Inverted Hammer volume with recent average
        recent_avg = candles.iloc[-4:-1]['volume'].mean()
        current_volume = candles.iloc[-1]['volume']
        
        return current_volume > recent_avg * 1.2  # 20% volume increase
    
    def _create_pattern_result(self, candles, inverted_hammer, confidence):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        inverted_hammer: pandas.Series
            The Inverted Hammer candle
        confidence: float
            Pattern confidence score
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Trading parameters
        entry_price = inverted_hammer['high'] * 1.001  # Entry above the high
        stop_price = inverted_hammer['low'] * 0.999  # Stop below the low
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2)  # 2:1 reward-risk
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'upper_wick': self.calculate_upper_wick(inverted_hammer),
                'lower_wick': self.calculate_lower_wick(inverted_hammer),
                'body_size': self.calculate_body_size(inverted_hammer)
            },
            'notes': f"Inverted Hammer at ${inverted_hammer['low']:.2f} suggesting potential reversal after downtrend"
        }
        
        self.log_detection(result)
        return result