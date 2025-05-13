"""
Tweezer Bottom Pattern

A bullish reversal pattern that consists of two consecutive candles where:
- The first candle is bearish (red) in a downtrend
- The second candle is bullish (green)
- Both candles have similar lows (the tweezer part)
"""

from ..base_pattern import BasePattern


class TweezerBottomPattern(BasePattern):
    """
    Detects the Tweezer Bottom candlestick pattern.
    
    A Tweezer Bottom consists of:
    - First bearish candle in a downtrend
    - Second bullish candle with a similar low price
    - Represents a potential reversal from bearish to bullish
    """
    
    def __init__(self):
        super().__init__(
            name="Tweezer Bottom",
            pattern_type="double",
            min_candles_required=7  # Need context
        )
        
        # Pattern parameters
        self.max_low_difference_pct = 0.1  # Maximum percentage difference between lows
    
    def detect(self, candles):
        """
        Detect Tweezer Bottom pattern in candlestick data.
        
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
        
        # Check Tweezer Bottom criteria
        
        # 1. First candle should be bearish
        if not self.is_bearish_candle(first_candle):
            return None
            
        # 2. Second candle should be bullish
        if not self.is_bullish_candle(second_candle):
            return None
            
        # 3. Both candles should have similar lows (the tweezer part)
        low_difference = abs(second_candle['low'] - first_candle['low'])
        avg_low = (second_candle['low'] + first_candle['low']) / 2
        
        # Calculate percentage difference
        low_difference_pct = (low_difference / avg_low) * 100
        
        if low_difference_pct > self.max_low_difference_pct:
            return None
            
        # 4. Check if we're in a downtrend
        trend = self.calculate_trend(candles.iloc[:-2])
        
        if trend != 'downtrend':
            # Less confident signal if not in a downtrend
            confidence_adjustment = -15
        else:
            confidence_adjustment = 0
            
        # Calculate pattern score
        pattern_score = self._calculate_pattern_score(first_candle, second_candle, low_difference_pct)
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score + confidence_adjustment,
            trend_alignment=(trend == 'downtrend'),
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(candles, first_candle, second_candle, confidence, trend)
        
    def _calculate_pattern_score(self, first_candle, second_candle, low_difference_pct):
        """Calculate pattern strength based on ideal proportions."""
        score = 60  # Base score
        
        # 1. Similar lows (more similar is better)
        if low_difference_pct < 0.05:
            score += 20  # Very similar lows
        elif low_difference_pct < 0.1:
            score += 10
            
        # 2. Strong bearish first candle
        first_body_size = self.calculate_body_size(first_candle)
        first_range = self.calculate_range(first_candle)
        
        if first_range > 0:
            first_body_ratio = first_body_size / first_range
            if first_body_ratio > 0.7:
                score += 10  # Strong bearish candle
                
        # 3. Strong bullish second candle
        second_body_size = self.calculate_body_size(second_candle)
        second_range = self.calculate_range(second_candle)
        
        if second_range > 0:
            second_body_ratio = second_body_size / second_range
            if second_body_ratio > 0.7:
                score += 10  # Strong bullish candle
                
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
        # Entry is above the high of the second candle
        entry_price = second_candle['high'] * 1.001
        
        # Stop loss is below the lows of both candles
        stop_price = min(first_candle['low'], second_candle['low']) * 0.999
        
        # Target based on risk-reward
        risk = entry_price - stop_price
        target_price = entry_price + (risk * 2)  # 2:1 reward-risk
        
        # Direction is bullish as this is a reversal pattern
        direction = "bullish"
        
        # Context description
        if trend == 'downtrend':
            context = "strong reversal signal at support"
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
                'low_difference_pct': abs(second_candle['low'] - first_candle['low']) / 
                                  ((second_candle['low'] + first_candle['low']) / 2) * 100
            },
            'notes': f"Tweezer Bottom at ${second_candle['close']:.2f}, {context}"
        }
        
        self.log_detection(result)
        return result