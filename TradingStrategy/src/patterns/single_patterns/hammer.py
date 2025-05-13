"""
Hammer Pattern

A bullish reversal pattern that forms at the bottom of a downtrend.
Characterized by a small body at the top of the candle's range and a long lower wick.
"""

from ..base_pattern import BasePattern
import numpy as np


class HammerPattern(BasePattern):
    """
    Detects the Hammer candlestick pattern.
    
    A hammer has:
    - Small body at the upper end of the price range
    - Long lower shadow (at least 2x the body size)
    - Little or no upper shadow
    - Appears after a downtrend
    """
    
    def __init__(self):
        super().__init__(
            name="Hammer",
            pattern_type="single",
            min_candles_required=10  # Need enough candles to determine trend
        )
    
    def detect(self, candles):
        """
        Detect hammer pattern in candlestick data.
        
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
        
        # Look at the most recent candle
        current_candle = candles.iloc[-1]
        
        # Determine the trend leading up to this candle
        trend = self.calculate_trend(candles.iloc[:-1], lookback=10)
        
        # Hammer should appear in a downtrend
        if trend != 'downtrend':
            self.logger.debug("Not in a downtrend, hammer pattern invalid")
            return None
        
        # Calculate candle characteristics
        body_size = self.calculate_body_size(current_candle)
        upper_wick = self.calculate_upper_wick(current_candle)
        lower_wick = self.calculate_lower_wick(current_candle)
        total_range = self.calculate_range(current_candle)
        
        # Hammer criteria:
        # 1. Body should be in the upper third of the range
        body_position = (min(current_candle['open'], current_candle['close']) - current_candle['low']) / total_range
        if body_position < 0.66:  # Body must be in upper 1/3
            self.logger.debug("Body not in upper third of range")
            return None
        
        # 2. Lower wick should be at least 2x the body size
        if lower_wick < (2 * body_size):
            self.logger.debug("Lower wick not long enough")
            return None
        
        # 3. Upper wick should be small (less than body size)
        if upper_wick > body_size:
            self.logger.debug("Upper wick too long")
            return None
        
        # 4. Body should be relatively small compared to the total range
        if body_size > (total_range * 0.35):
            self.logger.debug("Body too large relative to range")
            return None
        
        # Calculate pattern strength/confidence
        pattern_score = self._calculate_pattern_score(body_size, upper_wick, lower_wick, total_range)
        
        # Check volume confirmation
        volume_trend = self.calculate_volume_trend(candles, lookback=5)
        volume_confirmation = volume_trend in ['increasing', 'stable']
        
        # Calculate final confidence
        confidence = self.calculate_confidence(
            pattern_score=pattern_score,
            trend_alignment=True,  # Pattern appears in appropriate trend
            volume_confirmation=volume_confirmation
        )
        
        # Prepare result
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': current_candle['high'] + (total_range * 0.01),  # Entry above the high
            'stop_price': current_candle['low'] - (total_range * 0.01),   # Stop below the low
            'target_price': current_candle['high'] + (2 * (current_candle['high'] - current_candle['low'])),  # 2:1 reward
            'candle_index': len(candles) - 1,
            'notes': f"Hammer pattern in {trend} with {confidence:.1f}% confidence"
        }
        
        self.log_detection(result)
        return result
    
    def _calculate_pattern_score(self, body_size, upper_wick, lower_wick, total_range):
        """
        Calculate the pattern score based on how well it matches ideal hammer proportions.
        
        Parameters:
        -----------
        body_size: float
            Size of the candle body
        upper_wick: float
            Size of upper wick
        lower_wick: float
            Size of lower wick
        total_range: float
            Total candle range
            
        Returns:
        --------
        float: Pattern score (0-100)
        """
        score = 50  # Base score
        
        # Ideal hammer proportions:
        # - Lower wick should be 2-3x the body
        # - Upper wick should be minimal
        # - Body should be small relative to range
        
        # Score based on lower wick proportion
        lower_wick_ratio = lower_wick / body_size if body_size > 0 else 0
        if 2 <= lower_wick_ratio <= 3:
            score += 20  # Perfect ratio
        elif 1.5 <= lower_wick_ratio < 2 or 3 < lower_wick_ratio <= 4:
            score += 10  # Good ratio
        
        # Score based on upper wick (smaller is better)
        upper_wick_ratio = upper_wick / body_size if body_size > 0 else 0
        if upper_wick_ratio < 0.1:
            score += 15  # Virtually no upper wick
        elif upper_wick_ratio < 0.3:
            score += 10  # Small upper wick
        elif upper_wick_ratio < 0.5:
            score += 5   # Acceptable upper wick
        
        # Score based on body size relative to range
        body_ratio = body_size / total_range if total_range > 0 else 0
        if body_ratio < 0.1:
            score += 15  # Very small body
        elif body_ratio < 0.2:
            score += 10  # Small body
        elif body_ratio < 0.3:
            score += 5   # Acceptable body
        
        return min(100, score)