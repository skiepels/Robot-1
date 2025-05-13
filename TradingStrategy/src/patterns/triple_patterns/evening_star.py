"""
Evening Star Pattern

A bearish reversal pattern consisting of three candles:
1. A large bullish candle
2. A small bodied candle (star)
3. A bearish candle closing at least halfway down the first candle
"""

from ..base_pattern import BasePattern


class EveningStarPattern(BasePattern):
    """
    Detects the Evening Star candlestick pattern.
    
    An Evening Star consists of:
    - A large bullish candle
    - A small bodied candle with a gap up (the star)
    - A bearish candle closing well into the first candle's body
    """
    
    def __init__(self):
        super().__init__(
            name="Evening Star",
            pattern_type="triple",
            min_candles_required=12  # Need context
        )
        
        # Pattern parameters
        self.max_star_body_ratio = 0.3  # Star body must be small
        self.min_reversal_ratio = 0.5   # Third candle must retrace at least 50% of first
    
    def detect(self, candles):
        """
        Detect Evening Star pattern in candlestick data.
        
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
        
        # Determine the trend leading up to this pattern
        trend = self.calculate_trend(candles.iloc[:-3])
        
        # Evening Star should appear in an uptrend
        if trend != 'uptrend':
            return None
        
        # Get the last three candles
        first_candle = candles.iloc[-3]  # Bullish candle
        star_candle = candles.iloc[-2]   # Star
        third_candle = candles.iloc[-1]  # Bearish candle
        
        # Check Evening Star criteria
        
        # 1. First candle should be bullish
        if not self.is_bullish_candle(first_candle):
            return None
        
        # 2. Second candle (star) should have a small body
        first_body_size = self.calculate_body_size(first_candle)
        star_body_size = self.calculate_body_size(star_candle)
        
        if star_body_size > first_body_size * self.max_star_body_ratio:
            return None
        
        # 3. Third candle should be bearish
        if not self.is_bearish_candle(third_candle):
            return None
        
        # 4. Third candle should close well into first candle's body
        first_body_low = first_candle['open']
        first_body_high = first_candle['close']
        
        reversal_ratio = (first_body_high - third_candle['close']) / (first_body_high - first_body_low)
        
        if reversal_ratio < self.min_reversal_ratio:
            return None
        
        # Pattern score based on quality of formation
        pattern_score = self._calculate_pattern_score(first_candle, star_candle, third_candle, reversal_ratio)
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=True,
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(
            candles, first_candle, star_candle, third_candle, confidence, reversal_ratio
        )
    
    def _calculate_pattern_score(self, first_candle, star_candle, third_candle, reversal_ratio):
        """Calculate pattern strength based on formation quality."""
        score = 50  # Base score
        
        # 1. Star should be very small
        first_body_size = self.calculate_body_size(first_candle)
        star_body_size = self.calculate_body_size(star_candle)
        
        star_ratio = star_body_size / first_body_size if first_body_size > 0 else 1
        
        if star_ratio < 0.1:
            score += 15  # Very small star
        elif star_ratio < 0.2:
            score += 10
        
        # 2. Ideally star should gap up from first candle
        first_body_high = max(first_candle['open'], first_candle['close'])
        star_body_low = min(star_candle['open'], star_candle['close'])
        
        if star_body_low > first_body_high:
            score += 10  # Clear gap up
        
        # 3. Third candle reversal should be strong
        if reversal_ratio > 0.8:
            score += 15  # Strong reversal
        elif reversal_ratio > 0.6:
            score += 10
        
        # 4. Third candle should be strong
        third_body_size = self.calculate_body_size(third_candle)
        third_range = self.calculate_range(third_candle)
        
        if third_range > 0:
            third_body_ratio = third_body_size / third_range
            if third_body_ratio > 0.7:
                score += 10  # Strong third candle
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume pattern confirms the Evening Star."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        # Ideal volume pattern: high on first, low on star, high on third
        first_volume = candles.iloc[-3]['volume']
        star_volume = candles.iloc[-2]['volume']
        third_volume = candles.iloc[-1]['volume']
        
        # Check if third candle has higher volume than star
        return third_volume > star_volume * 1.3  # 30% higher volume on confirmation
    
    def _create_pattern_result(self, candles, first_candle, star_candle, third_candle, 
                              confidence, reversal_ratio):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First bullish candle
        star_candle: pandas.Series
            Star candle
        third_candle: pandas.Series
            Third bearish candle
        confidence: float
            Pattern confidence score
        reversal_ratio: float
            Reversal ratio of third candle
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Trading parameters
        entry_price = third_candle['low'] * 0.999  # Entry below third candle
        stop_price = max(star_candle['high'], third_candle['high']) * 1.001  # Stop above pattern
        
        # Risk and target
        risk = stop_price - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk ratio
        
        # Ensure target is positive
        if target_price <= 0:
            target_price = entry_price * 0.95  # Default 5% down
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bearish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'reversal_ratio': reversal_ratio,
                'first_body_size': self.calculate_body_size(first_candle),
                'star_body_size': self.calculate_body_size(star_candle),
                'third_body_size': self.calculate_body_size(third_candle),
                'has_gap': min(star_candle['open'], star_candle['close']) > max(first_candle['open'], first_candle['close'])
            },
            'notes': f"Evening Star with {reversal_ratio:.1%} reversal, signaling potential trend reversal"
        }
        
        self.log_detection(result)
        return result