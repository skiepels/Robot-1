"""
Candle Over Candle Pattern

A simple but powerful entry signal where the current candle breaks above
the previous candle's high, indicating momentum continuation.
"""

from ..base_pattern import BasePattern
import numpy as np


class CandleOverCandlePattern(BasePattern):
    """
    Detects the Candle Over Candle pattern for entry timing.
    
    This pattern is often used as:
    1. A standalone entry signal in strong trends
    2. Confirmation for other patterns (Bull Flag, Micro Pullback)
    3. Re-entry signal after partial exits
    """
    
    def __init__(self):
        super().__init__(
            name="Candle Over Candle",
            pattern_type="single",  # Though it uses 2 candles, it's a simple pattern
            min_candles_required=5   # Need some context
        )
        
        # Pattern parameters
        self.min_trend_strength = 0.5  # Minimum trend strength required
        self.require_bullish_candle = True  # Entry candle must be green
        self.min_volume_increase = 1.1  # 10% volume increase preferred
    
    def detect(self, candles):
        """
        Detect Candle Over Candle pattern in candlestick data.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            OHLCV candlestick data with indicators
            
        Returns:
        --------
        dict or None: Pattern detection result
        """
        # Validate candlestick data
        if not self.validate_candles(candles):
            return None
        
        if len(candles) < self.min_candles_required:
            return None
        
        # Get the last two candles
        prev_candle = candles.iloc[-2]
        current_candle = candles.iloc[-1]
        
        # Basic pattern criteria
        if not self._is_candle_over_candle(prev_candle, current_candle):
            return None
        
        # Verify market context
        if not self._verify_market_context(candles):
            return None
        
        # Create pattern result
        return self._create_pattern_result(candles, prev_candle, current_candle)
    
    def _is_candle_over_candle(self, prev_candle, current_candle):
        """
        Check if current candle breaks above previous candle's high.
        
        Parameters:
        -----------
        prev_candle: pandas.Series
            Previous candle data
        current_candle: pandas.Series
            Current candle data
            
        Returns:
        --------
        bool: True if pattern is valid
        """
        # Current high must exceed previous high
        if current_candle['high'] <= prev_candle['high']:
            return False
        
        # Current candle should be bullish (if required)
        if self.require_bullish_candle and not self.is_bullish_candle(current_candle):
            return False
        
        # Current close should be in upper half of range (strong close)
        candle_range = current_candle['high'] - current_candle['low']
        if candle_range > 0:
            close_position = (current_candle['close'] - current_candle['low']) / candle_range
            if close_position < 0.5:  # Weak close
                return False
        
        return True
    
    def _verify_market_context(self, candles):
        """
        Verify the market context is suitable for this pattern.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
            
        Returns:
        --------
        bool: True if context is valid
        """
        # Check overall trend
        trend = self.calculate_trend(candles)
        if trend != 'uptrend':
            return False
        
        # Check trend strength
        trend_strength = self._calculate_trend_strength(candles)
        if trend_strength < self.min_trend_strength:
            return False
        
        # Verify price is above key moving averages
        current_candle = candles.iloc[-1]
        
        if 'ema9' in current_candle and current_candle['close'] < current_candle['ema9']:
            return False
        
        if 'ema20' in current_candle and current_candle['close'] < current_candle['ema20']:
            return False
        
        # Check MACD is positive
        if 'macd_line' in current_candle and current_candle['macd_line'] <= 0:
            return False
        
        # Check volume pattern
        if not self._verify_volume_pattern(candles):
            return False
        
        return True
    
    def _verify_volume_pattern(self, candles):
        """
        Verify volume supports the breakout.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
            
        Returns:
        --------
        bool: True if volume pattern is valid
        """
        if 'volume' not in candles.columns:
            return True  # Can't verify, assume valid
        
        prev_candle = candles.iloc[-2]
        current_candle = candles.iloc[-1]
        
        # Current volume should be higher than previous
        if current_candle['volume'] < prev_candle['volume'] * self.min_volume_increase:
            return False
        
        # Current volume should be above recent average
        recent_avg_volume = candles['volume'].iloc[-10:-1].mean()
        if current_candle['volume'] < recent_avg_volume:
            return False
        
        return True
    
    def _create_pattern_result(self, candles, prev_candle, current_candle):
        """
        Create the pattern result with trading parameters.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        prev_candle: pandas.Series
            Previous candle
        current_candle: pandas.Series
            Current candle (breakout)
            
        Returns:
        --------
        dict: Complete pattern result
        """
        # Entry above current high
        entry_price = current_candle['high'] * 1.001
        
        # Stop loss options
        stop_option1 = prev_candle['low'] * 0.999  # Below previous candle
        stop_option2 = current_candle['low'] * 0.999  # Below current candle
        
        # Use the higher stop (more conservative)
        stop_price = max(stop_option1, stop_option2)
        
        # Target based on recent volatility
        recent_atr = self._calculate_atr(candles, period=14)
        target_price = entry_price + (recent_atr * 2)  # 2 ATR target
        
        # Ensure minimum 2:1 reward-risk
        risk = entry_price - stop_price
        min_target = entry_price + (risk * 2.0)
        target_price = max(target_price, min_target)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(candles, current_candle)
        
        # Check if this is part of a larger pattern
        larger_pattern = self._identify_larger_pattern(candles)
        
        result = {
            'pattern': self.name,
            'confidence': confidence,
            'direction': 'bullish',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'candle_index': len(candles) - 1,
            'pattern_data': {
                'prev_high': prev_candle['high'],
                'breakout_high': current_candle['high'],
                'breakout_close': current_candle['close'],
                'volume_increase': (current_candle['volume'] / prev_candle['volume'] 
                                  if 'volume' in candles.columns and prev_candle['volume'] > 0 
                                  else 1.0),
                'larger_pattern': larger_pattern
            },
            'notes': f"Candle over candle breakout at ${current_candle['high']:.2f}"
        }
        
        if larger_pattern:
            result['notes'] += f" (part of {larger_pattern})"
        
        self.log_detection(result)
        return result
    
    def _calculate_trend_strength(self, candles):
        """
        Calculate trend strength on a 0-1 scale.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
            
        Returns:
        --------
        float: Trend strength score
        """
        if len(candles) < 10:
            return 0.5
        
        recent = candles.iloc[-10:]
        
        # Count bullish candles
        bullish_count = sum(1 for _, c in recent.iterrows() if self.is_bullish_candle(c))
        bullish_ratio = bullish_count / len(recent)
        
        # Check higher highs and higher lows
        highs = recent['high'].values
        lows = recent['low'].values
        
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        trend_consistency = (higher_highs + higher_lows) / (2 * (len(recent) - 1))
        
        # Check position relative to moving averages
        last_candle = candles.iloc[-1]
        ma_score = 0
        
        if 'ema9' in last_candle and last_candle['close'] > last_candle['ema9']:
            ma_score += 0.33
        if 'ema20' in last_candle and last_candle['close'] > last_candle['ema20']:
            ma_score += 0.33
        if 'ema200' in last_candle and last_candle['close'] > last_candle['ema200']:
            ma_score += 0.34
        
        # Weighted average
        trend_strength = (bullish_ratio * 0.3 + trend_consistency * 0.4 + ma_score * 0.3)
        
        return trend_strength
    
    def _calculate_atr(self, candles, period=14):
        """
        Calculate Average True Range.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        period: int
            ATR period
            
        Returns:
        --------
        float: ATR value
        """
        if len(candles) < period:
            period = len(candles)
        
        high = candles['high'].values[-period:]
        low = candles['low'].values[-period:]
        close = candles['close'].values[-period:]
        
        # Calculate True Range
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        # First TR is just high - low
        if len(high) > 0:
            tr_list.insert(0, high[0] - low[0])
        
        # Calculate ATR
        atr = np.mean(tr_list) if tr_list else 0
        
        return atr
    
    def _identify_larger_pattern(self, candles):
        """
        Check if this candle-over-candle is part of a larger pattern.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
            
        Returns:
        --------
        str or None: Name of larger pattern if identified
        """
        # This is a simplified check - in reality, you'd use the pattern detector
        
        # Check for potential Bull Flag
        if self._could_be_bull_flag_breakout(candles):
            return "Bull Flag"
        
        # Check for potential Micro Pullback
        if self._could_be_micro_pullback_entry(candles):
            return "Micro Pullback"
        
        # Check for potential First Pullback
        if self._could_be_first_pullback_entry(candles):
            return "First Pullback"
        
        return None
    
    def _could_be_bull_flag_breakout(self, candles):
        """Check if this could be a bull flag breakout."""
        if len(candles) < 10:
            return False
        
        # Look for pole and flag structure
        # Simplified check - look for strong move up followed by consolidation
        for i in range(len(candles) - 10, len(candles) - 5):
            # Check for potential pole
            pole_gain = (candles.iloc[i+3]['close'] - candles.iloc[i]['close']) / candles.iloc[i]['close'] * 100
            
            if pole_gain > 3:  # 3% minimum pole
                # Check for consolidation after
                consolidation_range = (candles.iloc[i+4:i+7]['high'].max() - 
                                     candles.iloc[i+4:i+7]['low'].min())
                avg_price = candles.iloc[i+4:i+7]['close'].mean()
                
                if consolidation_range / avg_price < 0.03:  # Tight consolidation
                    return True
        
        return False
    
    def _could_be_micro_pullback_entry(self, candles):
        """Check if this could be a micro pullback entry."""
        if len(candles) < 5:
            return False
        
        # Check for 1-3 candle pullback
        recent = candles.iloc[-5:]
        
        # Look for red candle(s) followed by green breakout
        for i in range(1, 4):
            if i >= len(recent) - 1:
                break
            
            potential_pullback = recent.iloc[-i-1]
            if self.is_bearish_candle(potential_pullback):
                # Check if pullback stayed above 9 EMA
                if 'ema9' in potential_pullback and potential_pullback['low'] > potential_pullback['ema9']:
                    return True
        
        return False
    
    def _could_be_first_pullback_entry(self, candles):
        """Check if this could be a first pullback entry."""
        if len(candles) < 10:
            return False
        
        # Look for initial surge followed by first pullback
        for i in range(3, 8):
            if i >= len(candles) - 2:
                break
            
            # Check for surge
            surge_start = candles.iloc[-i-3:-i]
            surge_gain = (surge_start.iloc[-1]['close'] - surge_start.iloc[0]['close']) / surge_start.iloc[0]['close'] * 100
            
            if surge_gain > 2:  # 2% minimum surge
                # Check for pullback after surge
                pullback_candles = candles.iloc[-i:-1]
                red_count = sum(1 for _, c in pullback_candles.iterrows() if self.is_bearish_candle(c))
                
                if red_count > 0 and red_count < len(pullback_candles):
                    return True
        
        return False
    
    def _calculate_pattern_confidence(self, candles, current_candle):
        """Calculate confidence score for the pattern."""
        base_confidence = 60  # Base confidence for simple pattern
        
        # Strong trend increases confidence
        trend_strength = self._calculate_trend_strength(candles)
        if trend_strength > 0.8:
            base_confidence += 15
        elif trend_strength > 0.6:
            base_confidence += 10
        elif trend_strength > 0.5:
            base_confidence += 5
        
        # Volume confirmation
        if 'volume' in candles.columns:
            prev_candle = candles.iloc[-2]
            volume_ratio = current_candle['volume'] / prev_candle['volume'] if prev_candle['volume'] > 0 else 1
            
            if volume_ratio > 2.0:
                base_confidence += 15
            elif volume_ratio > 1.5:
                base_confidence += 10
            elif volume_ratio > 1.2:
                base_confidence += 5
        
        # Strong close increases confidence
        candle_range = current_candle['high'] - current_candle['low']
        if candle_range > 0:
            close_position = (current_candle['close'] - current_candle['low']) / candle_range
            if close_position > 0.8:
                base_confidence += 5
        
        # Part of larger pattern increases confidence
        if self._identify_larger_pattern(candles):
            base_confidence += 10
        
        return min(100, base_confidence)