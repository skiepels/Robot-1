"""
Tweezer Top Pattern

A bearish reversal pattern that consists of two consecutive candles where:
- The first candle is bullish (green) in an uptrend
- The second candle is bearish (red)
- Both candles have similar highs (the tweezer part)
"""

from ..base_pattern import BasePattern


class TweezerTopPattern(BasePattern):
    """
    Detects the Tweezer Top candlestick pattern.
    
    A Tweezer Top consists of:
    - First bullish candle in an uptrend
    - Second bearish candle with a similar high price
    - Represents a potential reversal from bullish to bearish
    """
    
    def __init__(self):
        super().__init__(
            name="Tweezer Top",
            pattern_type="double",
            min_candles_required=7  # Need context
        )
        
        # Pattern parameters
        self.max_high_difference_pct = 0.1  # Maximum percentage difference between highs
    
    def detect(self, candles):
        """
        Detect Tweezer Top pattern in candlestick data.
        
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
        
        # Look at the last two candles
        first_candle = candles.iloc[-2]
        second_candle = candles.iloc[-1]
        
        # Check Tweezer Top criteria
        
        # 1. First candle should be bullish
        if not self.is_bullish_candle(first_candle):
            return None
            
        # 2. Second candle should be bearish
        if not self.is_bearish_candle(second_candle):
            return None
            
        # 3. Both candles should have similar highs (the tweezer part)
        high_difference = abs(second_candle['high'] - first_candle['high'])
        avg_high = (second_candle['high'] + first_candle['high']) / 2
        
        # Calculate percentage difference
        high_difference_pct = (high_difference / avg_high) * 100
        
        if high_difference_pct > self.max_high_difference_pct:
            return None
            
        # 4. Check if we're in an uptrend
        trend = self.calculate_trend(candles.iloc[:-2])
        
        if trend != 'uptrend':
            # Less confident signal if not in an uptrend
            confidence_adjustment = -15
        else:
            confidence_adjustment = 0
            
        # Calculate pattern score
        pattern_score = self._calculate_pattern_score(first_candle, second_candle, high_difference_pct)
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score + confidence_adjustment,
            trend_alignment=(trend == 'uptrend'),
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(candles, first_candle, second_candle, confidence, trend)
        
    def _calculate_pattern_score(self, first_candle, second_candle, high_difference_pct):
        """Calculate pattern strength based on ideal proportions."""
        score = 60  # Base score
        
        # 1. Similar highs (more similar is better)
        if high_difference_pct < 0.05:
            score += 20  # Very similar highs
        elif high_difference_pct < 0.1:
            score += 10
            
        # 2. Strong bullish first candle
        first_body_size = self.calculate_body_size(first_candle)
        first_range = self.calculate_range(first_candle)
        
        if first_range > 0:
            first_body_ratio = first_body_size / first_range
            if first_body_ratio > 0.7:
                score += 10  # Strong bullish candle
                
        # 3. Strong bearish second candle
        second_body_size = self.calculate_body_size(second_candle)
        second_range = self.calculate_range(second_candle)
        
        if second_range > 0:
            second_body_ratio = second_body_size / second_range
            if second_body_ratio > 0.7:
                score += 10  # Strong bearish candle
                
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 2:
            return False
            
        first_volume = candles.iloc[-2]['volume']
        second_volume = candles.iloc[-1]['volume']
        
        # Second candle should have higher volume for confirmation
        return second_volume > first_volume * 1.2
    
    def _create_pattern_result(self, candles, first_candle, second_candle, confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First candle in pattern
        second_candle: pandas.Series
            Second candle in pattern
        confidence: float
            Pattern confidence score
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Entry is below the low of the second candle
        entry_price = second_candle['low'] * 0.999
        
        # Stop loss is above the highs of both candles
        stop_price = max(first_candle['high'], second_candle['high']) * 1.001
        
        # Target based on risk-reward
        risk = stop_price - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk
        
        # Ensure target is positive
        if target_price <= 0:
            target_price = entry_price * 0.95  # Default 5% down
        
        # Direction is bearish as this is a reversal pattern
        direction = "bearish"
        
        # Context description
        if trend == 'uptrend':
            context = "strong reversal signal at resistance"
        else:
            context = "potential reversal signal"
            
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'first_candle_body': self.calculate_body_size(first_candle),
                'second_candle_body': self.calculate_body_size(second_candle),
                'high_difference_pct': abs(second_candle['high'] - first_candle['high']) / 
                                   ((second_candle['high'] + first_candle['high']) / 2) * 100
            },
            'notes': f"Tweezer Top at ${second_candle['close']:.2f}, {context}"
        }
        
        self.log_detection(result)
        return result