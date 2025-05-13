"""
Doji Pattern

A single candlestick pattern where the open and close prices are nearly equal,
creating a cross or plus sign. Represents market indecision.
"""

from ..base_pattern import BasePattern


class DojiPattern(BasePattern):
    """
    Detects the Doji candlestick pattern.
    
    A Doji forms when the open and close prices are virtually equal,
    creating a candle with a very small body and wicks on both sides.
    It represents indecision in the market.
    """
    
    def __init__(self):
        super().__init__(
            name="Doji",
            pattern_type="single",
            min_candles_required=5  # Need context
        )
        
        # Pattern parameters
        self.body_threshold = 0.1  # Maximum body to range ratio
    
    def detect(self, candles):
        """
        Detect Doji pattern in candlestick data.
        
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
        
        # Calculate body and range
        body_size = self.calculate_body_size(current_candle)
        candle_range = self.calculate_range(current_candle)
        
        # Check for division by zero
        if candle_range == 0:
            return None
        
        # Calculate body to range ratio
        body_to_range_ratio = body_size / candle_range
        
        # Check if this is a Doji
        if body_to_range_ratio <= self.body_threshold:
            # Determine trend for context
            trend = self.calculate_trend(candles.iloc[:-1])
            
            # Create pattern result
            return self._create_pattern_result(candles, current_candle, body_to_range_ratio, trend)
        
        return None
    
    def _create_pattern_result(self, candles, doji_candle, body_ratio, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        doji_candle: pandas.Series
            The Doji candle
        body_ratio: float
            Body to range ratio
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Doji is a neutral pattern, but context matters
        direction = "neutral"
        
        # In an uptrend, Doji can be a warning sign
        if trend == "uptrend":
            direction = "potentially_bearish"
            
        # In a downtrend, Doji can signal potential reversal
        elif trend == "downtrend":
            direction = "potentially_bullish"
        
        # Calculate confidence based on how "perfect" the Doji is
        confidence = 100 - (body_ratio / self.body_threshold) * 100
        
        # For Doji, we don't provide entry/exit points directly
        # but rather a warning or confirmation signal
        notes = f"Doji indicating market indecision in {trend}. "
        
        if direction == "potentially_bearish":
            notes += "This may signal exhaustion in the uptrend."
        elif direction == "potentially_bullish":
            notes += "This may signal potential trend reversal."
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': direction,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'body_ratio': body_ratio,
                'upper_wick': self.calculate_upper_wick(doji_candle),
                'lower_wick': self.calculate_lower_wick(doji_candle),
                'trend_context': trend
            },
            'notes': notes
        }
        
        # Doji is more of a warning signal, so we don't include trade parameters
        # unless it's a specific type of Doji in a clear context
        
        self.log_detection(result)
        return result