"""
Evening Doji Star Pattern

A bearish reversal pattern that consists of three candles:
1. A large bullish candle in an uptrend
2. A doji candle that gaps up
3. A bearish candle that closes well into the body of the first candle
"""

from ..base_pattern import BasePattern


class EveningDojiStarPattern(BasePattern):
    """
    Detects the Evening Doji Star candlestick pattern.
    
    The Evening Doji Star pattern consists of:
    - A large bullish candle in an uptrend
    - A doji candle (small body) with a gap up from the first candle
    - A bearish candle that closes well into the body of the first candle
    - Indicates a potentially strong bearish reversal
    """
    
    def __init__(self):
        super().__init__(
            name="Evening Doji Star",
            pattern_type="triple",
            min_candles_required=10  # Need context
        )
        
        # Pattern parameters
        self.doji_body_threshold = 0.1  # Maximum body size for doji relative to range
        self.min_first_candle_body = 0.5  # Minimum body size of first candle relative to range
        self.min_reversal_ratio = 0.5  # Third candle must retrace at least 50% of first candle's body
        self.min_gap_percent = 0.1  # Minimum gap between first candle and doji
    
    def detect(self, candles):
        """
        Detect Evening Doji Star pattern in candlestick data.
        
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
        
        # Get the three candles for the pattern
        first_candle = candles.iloc[-3]
        doji_candle = candles.iloc[-2]
        third_candle = candles.iloc[-1]
        
        # Check Evening Doji Star criteria
        
        # 1. First candle must be bullish with a substantial body
        if not self.is_bullish_candle(first_candle):
            return None
            
        first_range = self.calculate_range(first_candle)
        first_body = self.calculate_body_size(first_candle)
        
        if first_range == 0 or (first_body / first_range) < self.min_first_candle_body:
            return None
        
        # 2. Second candle must be a doji (very small body)
        doji_range = self.calculate_range(doji_candle)
        doji_body = self.calculate_body_size(doji_candle)
        
        if doji_range == 0 or (doji_body / doji_range) > self.doji_body_threshold:
            return None
        
        # 3. Check for gap up between first candle and doji
        first_body_high = first_candle['close']  # First candle is bullish
        doji_body_low = min(doji_candle['open'], doji_candle['close'])
        
        # Calculate gap as percentage
        gap_percent = (doji_body_low - first_body_high) / first_body_high * 100
        
        if gap_percent < self.min_gap_percent:
            # Allow some flexibility if there's a clear separation
            if doji_body_low <= first_body_high:
                return None
        
        # 4. Third candle must be bearish with significant penetration into first candle
        if not self.is_bearish_candle(third_candle):
            return None
            
        # Calculate penetration ratio
        first_body_low = first_candle['open']  # First candle is bullish
        first_body_size = first_body_high - first_body_low
        
        penetration = first_body_high - third_candle['close']
        penetration_ratio = penetration / first_body_size if first_body_size > 0 else 0
        
        if penetration_ratio < self.min_reversal_ratio:
            return None
        
        # 5. Determine if we're in an uptrend (for reversal context)
        trend = self.calculate_trend(candles.iloc[:-3])
        
        if trend != 'uptrend':
            # Less confident signal if not in an uptrend
            confidence_adjustment = -15
        else:
            confidence_adjustment = 0
        
        # Calculate pattern strength
        pattern_score = self._calculate_pattern_score(
            first_candle, doji_candle, third_candle, gap_percent, penetration_ratio
        )
        
        # Check volume pattern
        volume_confirmed = self._check_volume_pattern(candles)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score + confidence_adjustment,
            trend_alignment=(trend == 'uptrend'),
            volume_confirmation=volume_confirmed
        )
        
        # Create pattern result
        return self._create_pattern_result(
            candles, first_candle, doji_candle, third_candle, 
            gap_percent, penetration_ratio, confidence, trend
        )
    
    def _calculate_pattern_score(self, first_candle, doji_candle, third_candle, 
                                gap_percent, penetration_ratio):
        """Calculate pattern strength based on ideal proportions."""
        score = 65  # Base score
        
        # 1. Stronger doji (smaller body is better)
        doji_body_ratio = self.calculate_body_size(doji_candle) / self.calculate_range(doji_candle)
        
        if doji_body_ratio < 0.03:
            score += 10  # Perfect doji
        elif doji_body_ratio < 0.05:
            score += 5  # Good doji
        
        # 2. Gap size (larger gap is better, up to a point)
        if gap_percent > 1.0:
            score += 10  # Significant gap
        elif gap_percent > 0.5:
            score += 5  # Decent gap
        
        # 3. Penetration into first candle (deeper is better)
        if penetration_ratio > 0.8:
            score += 15  # Deep penetration
        elif penetration_ratio > 0.65:
            score += 10  # Good penetration
        elif penetration_ratio > 0.5:
            score += 5  # Adequate penetration
        
        # 4. Third candle strength
        third_body_ratio = self.calculate_body_size(third_candle) / self.calculate_range(third_candle)
        
        if third_body_ratio > 0.7:
            score += 10  # Strong bearish candle
        elif third_body_ratio > 0.5:
            score += 5  # Decent bearish candle
        
        return min(score, 100)
    
    def _check_volume_pattern(self, candles):
        """Check if volume supports the pattern."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        first_volume = candles.iloc[-3]['volume']
        doji_volume = candles.iloc[-2]['volume']
        third_volume = candles.iloc[-1]['volume']
        
        # Ideal volume pattern: Lower volume on doji, higher on first and third
        # Especially high on third candle for confirmation
        if doji_volume < first_volume and third_volume > doji_volume:
            return True
            
        # Alternative check - third candle should have above average volume
        if third_volume > (first_volume + doji_volume) / 2 * 1.2:
            return True
            
        return False
    
    def _create_pattern_result(self, candles, first_candle, doji_candle, third_candle, 
                             gap_percent, penetration_ratio, confidence, trend):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        first_candle: pandas.Series
            First bullish candle
        doji_candle: pandas.Series
            Middle doji candle
        third_candle: pandas.Series
            Last bearish candle
        gap_percent: float
            Gap percentage between first and doji
        penetration_ratio: float
            Penetration ratio of third candle into first
        confidence: float
            Pattern confidence score
        trend: str
            Current market trend
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Entry below the low of the third candle
        entry_price = third_candle['low'] * 0.999
        
        # Stop above the high of the pattern
        pattern_high = max(
            first_candle['high'],
            doji_candle['high'],
            third_candle['high']
        ) * 1.001
        
        # Target based on risk-reward
        risk = pattern_high - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk
        
        # Ensure target is positive
        if target_price <= 0:
            target_price = entry_price * 0.95  # Default 5% down
        
        # Pattern strength description
        strength_desc = "strong" if confidence >= 80 else "moderate" if confidence >= 70 else "potential"
        
        # Context description
        if trend == 'uptrend':
            context = "strong reversal signal at resistance"
        else:
            context = "reversal signal"
            
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bearish',
            'entry_price': entry_price,
            'stop_price': pattern_high,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'doji_body_ratio': self.calculate_body_size(doji_candle) / self.calculate_range(doji_candle),
                'gap_percent': gap_percent,
                'penetration_ratio': penetration_ratio,
                'third_candle_body_ratio': self.calculate_body_size(third_candle) / self.calculate_range(third_candle)
            },
            'notes': f"Evening Doji Star at ${third_candle['close']:.2f}, {strength_desc} {context}"
        }
        
        self.log_detection(result)
        return result