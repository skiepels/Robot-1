"""
Bullish Engulfing Pattern

A two-candlestick reversal pattern that forms in a downtrend, where the second
candle completely "engulfs" the body of the first candle.
"""

from ..base_pattern import BasePattern


class BullishEngulfingPattern(BasePattern):
    """
    Detects the Bullish Engulfing candlestick pattern.
    
    A Bullish Engulfing pattern consists of:
    - A bearish (red) candle in a downtrend
    - Followed by a bullish (green) candle that completely engulfs the body of the previous candle
    - Signals a potential trend reversal from bearish to bullish
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Engulfing",
            pattern_type="double",
            min_candles_required=10  # Need context
        )
    
    def detect(self, candles):
        """
        Detect Bullish Engulfing pattern in candlestick data.
        
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
        trend = self.calculate_trend(candles.iloc[:-2])
        
        # Bullish Engulfing should appear in a downtrend
        if trend != 'downtrend':
            return None
        
        # Get the last two candles
        first_candle = candles.iloc[-2]
        second_candle = candles.iloc[-1]
        
        # Check Bullish Engulfing criteria:
        # 1. First candle should be bearish (red)
        if not self.is_bearish_candle(first_candle):
            return None
        
        # 2. Second candle should be bullish (green)
        if not self.is_bullish_candle(second_candle):
            return None
        
        # 3. Second candle should engulf the body of the first candle
        if not self._is_bullish_engulfing(first_candle, second_candle):
            return None
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(first_candle, second_candle)
        
        # Check volume confirmation
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Create pattern result
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=True,
            volume_confirmation=volume_confirmed
        )
        
        return self._create_pattern_result(candles, first_candle, second_candle, confidence)
    
    def _is_bullish_engulfing(self, first_candle, second_candle):
        """
        Check if second candle engulfs the body of the first candle.
        
        Parameters:
        -----------
        first_candle: pandas.Series
            First candle in pattern
        second_candle: pandas.Series
            Second candle in pattern
            
        Returns:
        --------
        bool: True if pattern criteria are met
        """
        first_body_low = min(first_candle['open'], first_candle['close'])
        first_body_high = max(first_candle['open'], first_candle['close'])
        
        second_body_low = min(second_candle['open'], second_candle['close'])
        second_body_high = max(second_candle['open'], second_candle['close'])
        
        # Second candle body must completely engulf first candle body
        return (second_body_low <= first_body_low and 
                second_body_high >= first_body_high)
    
    def _calculate_pattern_score(self, first_candle, second_candle):
        """Calculate how strong the engulfing pattern is."""
        score = 50  # Base score
        
        # Size of engulfing candle matters
        first_body_size = self.calculate_body_size(first_candle)
        second_body_size = self.calculate_body_size(second_candle)
        
        # Engulfing candle should be significantly larger
        if second_body_size > first_body_size * 2:
            score += 20
        elif second_body_size > first_body_size * 1.5:
            score += 10
        
        # Second candle closing near its high is stronger
        second_range = self.calculate_range(second_candle)
        if second_range > 0:
            close_position = (second_candle['close'] - second_candle['low']) / second_range
            if close_position > 0.9:  # Very strong close
                score += 15
            elif close_position > 0.8:
                score += 10
            elif close_position > 0.7:
                score += 5
        
        # Complete engulfing (including shadows) is strongest
        if (second_candle['low'] <= first_candle['low'] and 
            second_candle['high'] >= first_candle['high']):
            score += 15
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume increased on the engulfing candle."""
        if 'volume' not in candles.columns or len(candles) < 2:
            return False
        
        # Compare engulfing volume with previous candle's volume
        first_volume = candles.iloc[-2]['volume']
        second_volume = candles.iloc[-1]['volume']
        
        return second_volume > first_volume * 1.5  # 50% volume increase
    
    def _create_pattern_result(self, candles, first_candle, second_candle, confidence):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First candle in pattern
        second_candle: pandas.Series
            Second (engulfing) candle
        confidence: float
            Pattern confidence score
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Trading parameters
        entry_price = second_candle['high'] * 1.001  # Entry above engulfing candle
        stop_price = min(first_candle['low'], second_candle['low']) * 0.999  # Stop below pattern
        
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
                'first_candle_size': self.calculate_body_size(first_candle),
                'second_candle_size': self.calculate_body_size(second_candle),
                'engulfing_ratio': self.calculate_body_size(second_candle) / 
                                 self.calculate_body_size(first_candle)
                                 if self.calculate_body_size(first_candle) > 0 else 0
            },
            'notes': f"Bullish Engulfing at ${second_candle['close']:.2f} signaling potential trend reversal"
        }
        
        self.log_detection(result)
        return result