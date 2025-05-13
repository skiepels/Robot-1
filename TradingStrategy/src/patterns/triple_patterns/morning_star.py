"""
Morning Star Pattern

A bullish reversal pattern consisting of three candles:
1. A large bearish candle
2. A small bodied candle (star)
3. A bullish candle closing at least halfway up the first candle
"""

from ..base_pattern import BasePattern


class MorningStarPattern(BasePattern):
    """
    Detects the Morning Star candlestick pattern.
    
    A Morning Star consists of:
    - A large bearish candle
    - A small bodied candle with a gap down (the star)
    - A bullish candle closing well into the first candle's body
    """
    
    def __init__(self):
        super().__init__(
            name="Morning Star",
            pattern_type="triple",
            min_candles_required=12  # Need context
        )
        
        # Pattern parameters
        self.max_star_body_ratio = 0.3  # Star body must be small
        self.min_recovery_ratio = 0.5   # Third candle must recover at least 50% of first
    
    def detect(self, candles):
        """
        Detect Morning Star pattern in candlestick data.
        
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
        
        # Morning Star should appear in a downtrend
        if trend != 'downtrend':
            return None
        
        # Get the last three candles
        first_candle = candles.iloc[-3]  # Bearish candle
        star_candle = candles.iloc[-2]   # Star
        third_candle = candles.iloc[-1]  # Bullish candle
        
        # Check Morning Star criteria
        
        # 1. First candle should be bearish
        if not self.is_bearish_candle(first_candle):
            return None
        
        # 2. Second candle (star) should have a small body
        first_body_size = self.calculate_body_size(first_candle)
        star_body_size = self.calculate_body_size(star_candle)
        
        if star_body_size > first_body_size * self.max_star_body_ratio:
            return None
        
        # 3. Third candle should be bullish
        if not self.is_bullish_candle(third_candle):
            return None
        
        # 4. Third candle should close well into first candle's body
        first_body_high = first_candle['open']
        first_body_low = first_candle['close']
        
        recovery_ratio = (third_candle['close'] - first_body_low) / (first_body_high - first_body_low)
        
        if recovery_ratio < self.min_recovery_ratio:
            return None
        
        # Pattern score based on quality of formation
        pattern_score = self._calculate_pattern_score(first_candle, star_candle, third_candle, recovery_ratio)
        
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
            candles, first_candle, star_candle, third_candle, confidence, recovery_ratio
        )
    
    def _calculate_pattern_score(self, first_candle, star_candle, third_candle, recovery_ratio):
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
        
        # 2. Ideally star should gap down from first candle
        first_body_low = min(first_candle['open'], first_candle['close'])
        star_body_high = max(star_candle['open'], star_candle['close'])
        
        if star_body_high < first_body_low:
            score += 10  # Clear gap down
        
        # 3. Third candle recovery should be strong
        if recovery_ratio > 0.8:
            score += 15  # Strong recovery
        elif recovery_ratio > 0.6:
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
        """Check if volume pattern confirms the Morning Star."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        # Ideal volume pattern: high on first, low on star, high on third
        first_volume = candles.iloc[-3]['volume']
        star_volume = candles.iloc[-2]['volume']
        third_volume = candles.iloc[-1]['volume']
        
        # Check if third candle has higher volume than star
        return third_volume > star_volume * 1.3  # 30% higher volume on confirmation
    
    def _create_pattern_result(self, candles, first_candle, star_candle, third_candle, 
                              confidence, recovery_ratio):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First bearish candle
        star_candle: pandas.Series
            Star candle
        third_candle: pandas.Series
            Third bullish candle
        confidence: float
            Pattern confidence score
        recovery_ratio: float
            Recovery ratio of third candle
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Trading parameters
        entry_price = third_candle['high'] * 1.001  # Entry above third candle
        stop_price = min(star_candle['low'], third_candle['low']) * 0.999  # Stop below pattern
        
        # Risk and target
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2)  # 2:1 reward-risk ratio
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'recovery_ratio': recovery_ratio,
                'first_body_size': self.calculate_body_size(first_candle),
                'star_body_size': self.calculate_body_size(star_candle),
                'third_body_size': self.calculate_body_size(third_candle),
                'has_gap': max(star_candle['open'], star_candle['close']) < min(first_candle['open'], first_candle['close'])
            },
            'notes': f"Morning Star with {recovery_ratio:.1%} recovery, signaling potential trend reversal"
        }
        
        self.log_detection(result)
        return result