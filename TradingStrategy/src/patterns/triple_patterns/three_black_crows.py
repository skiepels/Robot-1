"""
Three Black Crows Pattern

A bearish reversal pattern consisting of three consecutive bearish (red) candles,
each closing lower than the previous, often appearing after an uptrend.
"""

from ..base_pattern import BasePattern
import numpy as np


class ThreeBlackCrowsPattern(BasePattern):
    """
    Detects the Three Black Crows candlestick pattern.
    
    The Three Black Crows pattern consists of:
    - Three consecutive bearish (red) candles
    - Each candle opens within the body of the previous candle
    - Each candle closes at or near its low
    - Each candle closes lower than the previous
    - Appears after an uptrend as a strong reversal signal
    """
    
    def __init__(self):
        super().__init__(
            name="Three Black Crows",
            pattern_type="triple",
            min_candles_required=10  # Need context
        )
        
        # Pattern parameters
        self.min_body_size_ratio = 0.5  # Minimum body to range ratio
        self.max_lower_shadow_ratio = 0.25  # Maximum lower shadow relative to body
        self.min_close_progress = 0.2  # Minimum percentage progress of each close
    
    def detect(self, candles):
        """
        Detect Three Black Crows pattern in candlestick data.
        
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
        
        # Get the last three candles
        first_candle = candles.iloc[-3]
        second_candle = candles.iloc[-2]
        third_candle = candles.iloc[-1]
        
        # Check Three Black Crows criteria
        
        # 1. All three candles must be bearish
        if not (self.is_bearish_candle(first_candle) and 
                self.is_bearish_candle(second_candle) and 
                self.is_bearish_candle(third_candle)):
            return None
        
        # 2. Each candle must close lower than the previous
        if not (second_candle['close'] < first_candle['close'] and
                third_candle['close'] < second_candle['close']):
            return None
        
        # 3. Each candle should have a decent body size
        for candle in [first_candle, second_candle, third_candle]:
            body_size = self.calculate_body_size(candle)
            total_range = self.calculate_range(candle)
            
            if total_range == 0 or (body_size / total_range) < self.min_body_size_ratio:
                return None
        
        # 4. Small lower shadows (closes near the lows)
        for candle in [first_candle, second_candle, third_candle]:
            lower_shadow = self.calculate_lower_wick(candle)
            body_size = self.calculate_body_size(candle)
            
            if body_size == 0 or (lower_shadow / body_size) > self.max_lower_shadow_ratio:
                return None
        
        # 5. Each candle should open within the body of the previous candle
        if not (second_candle['open'] <= first_candle['open'] and 
                second_candle['open'] >= first_candle['close']):
            # Allow some flexibility
            if not (second_candle['open'] <= first_candle['open'] * 1.01 and 
                    second_candle['open'] >= first_candle['close'] * 0.99):
                return None
        
        if not (third_candle['open'] <= second_candle['open'] and 
                third_candle['open'] >= second_candle['close']):
            # Allow some flexibility
            if not (third_candle['open'] <= second_candle['open'] * 1.01 and 
                    third_candle['open'] >= second_candle['close'] * 0.99):
                return None
        
        # 6. Check sufficient progress in closing prices
        first_to_second_progress = first_candle['close'] - second_candle['close']
        second_to_third_progress = second_candle['close'] - third_candle['close']
        
        price_range = max(first_candle['high'], second_candle['high'], third_candle['high']) - \
                    min(first_candle['low'], second_candle['low'], third_candle['low'])
        
        if price_range == 0:
            return None
            
        if (first_to_second_progress / price_range < self.min_close_progress or
            second_to_third_progress / price_range < self.min_close_progress):
            return None
        
        # 7. Determine if we're in an uptrend (for reversal context)
        trend = self.calculate_trend(candles.iloc[:-3])
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(
            first_candle, second_candle, third_candle, trend
        )
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=(trend == 'uptrend'),  # Most significant as reversal
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(
            candles, [first_candle, second_candle, third_candle], 
            confidence, trend
        )
    
    def _calculate_pattern_score(self, first_candle, second_candle, third_candle, trend):
        """Calculate pattern strength based on ideal proportions."""
        score = 65  # Base score
        
        # 1. Trend context
        if trend == 'uptrend':
            score += 10  # Ideal context for reversal
        
        # 2. Strong bodies (larger is better)
        avg_body_ratio = 0
        for candle in [first_candle, second_candle, third_candle]:
            body_size = self.calculate_body_size(candle)
            total_range = self.calculate_range(candle)
            
            if total_range > 0:
                avg_body_ratio += body_size / total_range
                
        avg_body_ratio /= 3
        
        if avg_body_ratio > 0.8:
            score += 15  # Very strong bodies
        elif avg_body_ratio > 0.6:
            score += 10
        
        # 3. Progressive strength (each candle stronger than the last)
        if (self.calculate_body_size(second_candle) >= self.calculate_body_size(first_candle) and
            self.calculate_body_size(third_candle) >= self.calculate_body_size(second_candle)):
            score += 10  # Progressively stronger
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        first_volume = candles.iloc[-3]['volume']
        second_volume = candles.iloc[-2]['volume']
        third_volume = candles.iloc[-1]['volume']
        
        # Ideal volume pattern: Each candle has higher volume than the previous
        # At minimum, the third candle should have higher volume
        if third_volume > second_volume and second_volume > first_volume:
            return True
        
        # Or at least increasing volume trend
        avg_previous_volume = candles.iloc[-6:-3]['volume'].mean() if len(candles) >= 6 else first_volume
        avg_pattern_volume = (first_volume + second_volume + third_volume) / 3
        
        return avg_pattern_volume > avg_previous_volume * 1.2
    
    def _create_pattern_result(self, candles, pattern_candles, confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        pattern_candles: list
            The three candles forming the pattern
        confidence: float
            Pattern confidence score
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Get the third candle
        third_candle = pattern_candles[2]
        
        # Entry below the low of the third candle
        entry_price = third_candle['low'] * 0.999
        
        # Stop above the high of the pattern
        stop_price = max(
            pattern_candles[0]['high'],
            pattern_candles[1]['high'],
            pattern_candles[2]['high']
        ) * 1.001
        
        # Target based on risk-reward
        risk = stop_price - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk
        
        # Ensure target is positive
        if target_price <= 0:
            target_price = entry_price * 0.95  # Default 5% down
        
        # Pattern strength description
        strength = "strong"
        if confidence < 70:
            strength = "moderate"
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bearish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'first_candle_body': self.calculate_body_size(pattern_candles[0]),
                'second_candle_body': self.calculate_body_size(pattern_candles[1]),
                'third_candle_body': self.calculate_body_size(pattern_candles[2]),
                'total_decline': pattern_candles[0]['open'] - pattern_candles[2]['close']
            },
            'notes': f"Three Black Crows forming a {strength} bearish reversal signal"
        }
        
        self.log_detection(result)
        return result