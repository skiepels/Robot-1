"""
Shooting Star Pattern

A bearish reversal pattern that forms at the top of an uptrend.
Characterized by a small body at the bottom of the candle's range with a long upper wick.
"""

from ..base_pattern import BasePattern


class ShootingStarPattern(BasePattern):
    """
    Detects the Shooting Star candlestick pattern.
    
    A Shooting Star is characterized by:
    - Small body at lower end of the price range
    - Long upper shadow (at least 2x the body size)
    - Little or no lower shadow
    - Appears after an uptrend
    """
    
    def __init__(self):
        super().__init__(
            name="Shooting Star",
            pattern_type="single",
            min_candles_required=5  # Need context
        )
        
        # Pattern parameters
        self.min_upper_wick_ratio = 2.0  # Upper wick should be at least 2x body
        self.max_lower_wick_ratio = 0.5  # Lower wick should be < 50% of body
        self.max_body_size_ratio = 0.3   # Body should be small relative to range
    
    def detect(self, candles):
        """
        Detect Shooting Star pattern in candlestick data.
        
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
        
        # Shooting star should appear in an uptrend
        if trend != 'uptrend':
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
        
        # Check Shooting Star criteria
        
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
        
        # Create pattern result
        return self._create_pattern_result(candles, current_candle, pattern_score)
    
    def _calculate_pattern_score(self, body_size, upper_wick, lower_wick, total_range):
        """Calculate how well the candle matches ideal Shooting Star proportions."""
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
    
    def _create_pattern_result(self, candles, shooting_star, pattern_score):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        shooting_star: pandas.Series
            The Shooting Star candle
        pattern_score: float
            Score indicating pattern quality
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Context analysis
        trend_strength = self._calculate_trend_strength(candles)
        volume_increasing = self._check_volume_increase(candles)
        
        # Calculate confidence score
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=True,  # Pattern appears in appropriate trend
            volume_confirmation=volume_increasing
        )
        
        # Trading parameters
        entry_price = shooting_star['low'] * 0.999  # Entry below the low
        stop_price = shooting_star['high'] * 1.001  # Stop above the high
        risk = stop_price - entry_price
        target_price = entry_price - (risk * 2)  # 2:1 reward-risk
        
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
                'upper_wick': self.calculate_upper_wick(shooting_star),
                'lower_wick': self.calculate_lower_wick(shooting_star),
                'body_size': self.calculate_body_size(shooting_star),
                'trend_strength': trend_strength
            },
            'notes': f"Shooting Star at ${shooting_star['high']:.2f} suggesting potential reversal"
        }
        
        self.log_detection(result)
        return result
    
    def _calculate_trend_strength(self, candles):
        """Calculate strength of the uptrend before the shooting star."""
        lookback = min(10, len(candles) - 1)
        trend_candles = candles.iloc[-lookback-1:-1]
        
        # Count bullish candles
        bullish_count = sum(1 for _, c in trend_candles.iterrows() if self.is_bullish_candle(c))
        bullish_ratio = bullish_count / len(trend_candles) if len(trend_candles) > 0 else 0
        
        # Calculate overall gain
        start_price = trend_candles.iloc[0]['close'] if len(trend_candles) > 0 else 0
        end_price = trend_candles.iloc[-1]['close'] if len(trend_candles) > 0 else 0
        
        if start_price > 0:
            gain_pct = ((end_price / start_price) - 1) * 100
        else:
            gain_pct = 0
        
        # Scale from 0-1
        return min(((bullish_ratio * 0.5) + (gain_pct / 20 * 0.5)), 1.0)
    
    def _check_volume_increase(self, candles):
        """Check if volume increased on the Shooting Star candle."""
        if 'volume' not in candles.columns or len(candles) < 3:
            return False
        
        # Compare shooting star volume with recent average
        recent_avg = candles.iloc[-4:-1]['volume'].mean()
        star_volume = candles.iloc[-1]['volume']
        
        return star_volume > recent_avg * 1.2  # 20% volume increase