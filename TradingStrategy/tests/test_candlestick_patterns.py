"""
Test Candlestick Patterns

Unit tests for the candlestick pattern recognition module based on Ross Cameron's approach.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.candlestick_patterns import CandlestickPatterns


class TestCandlestickPatterns(unittest.TestCase):
    """Test cases for the CandlestickPatterns class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pattern_detector = CandlestickPatterns()
        
        # Create a sample DataFrame with OHLCV data
        dates = pd.date_range(start='2025-05-01', periods=20, freq='1min')
        self.sample_data = pd.DataFrame({
            'open': [10.0, 10.2, 10.4, 10.1, 10.3, 10.5, 10.2, 10.4, 10.6, 10.3,
                   10.5, 10.7, 10.4, 10.6, 10.8, 10.5, 10.7, 10.9, 10.6, 10.8],
            'high': [10.3, 10.5, 10.7, 10.4, 10.6, 10.8, 10.5, 10.7, 10.9, 10.6,
                    10.8, 11.0, 10.7, 10.9, 11.1, 10.8, 11.0, 11.2, 10.9, 11.1],
            'low': [9.8, 10.0, 10.2, 9.9, 10.1, 10.3, 10.0, 10.2, 10.4, 10.1,
                  10.3, 10.5, 10.2, 10.4, 10.6, 10.3, 10.5, 10.7, 10.4, 10.6],
            'close': [10.2, 10.4, 10.1, 10.3, 10.5, 10.2, 10.4, 10.6, 10.3, 10.5,
                    10.7, 10.4, 10.6, 10.8, 10.5, 10.7, 10.9, 10.6, 10.8, 11.0],
            'volume': [5000, 6000, 4000, 5500, 7000, 4500, 5000, 6500, 5000, 5500,
                     7500, 5000, 6000, 8000, 5500, 6500, 9000, 6000, 7000, 10000]
        }, index=dates)
    
    def test_basic_candle_functions(self):
        """Test basic candle calculations."""
        # Test bullish/bearish candle identification
        self.assertTrue(self.pattern_detector.is_bullish_candle(10.0, 10.5))
        self.assertFalse(self.pattern_detector.is_bullish_candle(10.5, 10.0))
        
        self.assertTrue(self.pattern_detector.is_bearish_candle(10.5, 10.0))
        self.assertFalse(self.pattern_detector.is_bearish_candle(10.0, 10.5))
        
        # Test body size calculation
        self.assertEqual(self.pattern_detector.calculate_body_size(10.0, 10.5), 0.5)
        self.assertEqual(self.pattern_detector.calculate_body_size(10.5, 10.0), 0.5)
        
        # Test wick calculations
        self.assertEqual(self.pattern_detector.calculate_upper_wick(10.0, 10.5, 11.0), 0.5)
        self.assertEqual(self.pattern_detector.calculate_lower_wick(10.0, 10.5, 9.5), 0.5)
    
    def test_doji_detection(self):
        """Test doji candle pattern detection."""
        # Create a dataframe with a doji pattern
        doji_data = self.sample_data.copy()
        # Set a doji at index 5
        doji_data.loc[doji_data.index[5], 'open'] = 10.50
        doji_data.loc[doji_data.index[5], 'close'] = 10.51
        doji_data.loc[doji_data.index[5], 'high'] = 11.00
        doji_data.loc[doji_data.index[5], 'low'] = 10.00
        
        # Test doji detection
        self.assertTrue(self.pattern_detector.detect_doji(doji_data, 5))
        self.assertFalse(self.pattern_detector.detect_doji(doji_data, 0))  # Regular candle
    
    def test_hammer_detection(self):
        """Test hammer pattern detection."""
        # Create a dataframe with a hammer pattern
        hammer_data = self.sample_data.copy()
        # Set a downtrend before the hammer
        hammer_data.loc[hammer_data.index[5:8], 'close'] = [10.4, 10.2, 10.0]
        # Set a hammer at index 8
        hammer_data.loc[hammer_data.index[8], 'open'] = 10.05
        hammer_data.loc[hammer_data.index[8], 'close'] = 10.10
        hammer_data.loc[hammer_data.index[8], 'high'] = 10.15
        hammer_data.loc[hammer_data.index[8], 'low'] = 9.50
        
        # Test hammer detection
        self.assertTrue(self.pattern_detector.detect_hammer(hammer_data, 8))
        self.assertFalse(self.pattern_detector.detect_hammer(hammer_data, 0))  # Regular candle
    
    def test_shooting_star_detection(self):
        """Test shooting star pattern detection."""
        # Create a dataframe with a shooting star pattern
        star_data = self.sample_data.copy()
        # Set an uptrend before the shooting star
        star_data.loc[star_data.index[5:8], 'close'] = [10.0, 10.2, 10.4]
        # Set a shooting star at index 8
        star_data.loc[star_data.index[8], 'open'] = 10.45
        star_data.loc[star_data.index[8], 'close'] = 10.40
        star_data.loc[star_data.index[8], 'high'] = 11.00
        star_data.loc[star_data.index[8], 'low'] = 10.35
        
        # Test shooting star detection
        self.assertTrue(self.pattern_detector.detect_shooting_star(star_data, 8))
        self.assertFalse(self.pattern_detector.detect_shooting_star(star_data, 0))  # Regular candle
    
    def test_engulfing_patterns(self):
        """Test engulfing patterns detection."""
        # Create a dataframe with bullish and bearish engulfing patterns
        engulf_data = self.sample_data.copy()
        
        # Set a bearish candle followed by a bullish engulfing candle
        engulf_data.loc[engulf_data.index[5], 'open'] = 10.50
        engulf_data.loc[engulf_data.index[5], 'close'] = 10.30
        engulf_data.loc[engulf_data.index[6], 'open'] = 10.25
        engulf_data.loc[engulf_data.index[6], 'close'] = 10.55
        
        # Set a bullish candle followed by a bearish engulfing candle
        engulf_data.loc[engulf_data.index[10], 'open'] = 10.30
        engulf_data.loc[engulf_data.index[10], 'close'] = 10.50
        engulf_data.loc[engulf_data.index[11], 'open'] = 10.55
        engulf_data.loc[engulf_data.index[11], 'close'] = 10.25
        
        # Test engulfing patterns detection
        self.assertTrue(self.pattern_detector.detect_bullish_engulfing(engulf_data, 6))
        self.assertFalse(self.pattern_detector.detect_bullish_engulfing(engulf_data, 11))
        
        self.assertTrue(self.pattern_detector.detect_bearish_engulfing(engulf_data, 11))
        self.assertFalse(self.pattern_detector.detect_bearish_engulfing(engulf_data, 6))
    
    def test_bull_flag_detection(self):
        """Test bull flag pattern detection."""
        # Create a dataframe with a bull flag pattern
        flag_data = self.sample_data.copy()
        
        # Set a strong upward move (pole)
        flag_data.loc[flag_data.index[0:3], 'close'] = [10.0, 10.5, 11.0]
        flag_data.loc[flag_data.index[0:3], 'high'] = [10.2, 10.7, 11.2]
        
        # Set a consolidation with lower highs (flag)
        flag_data.loc[flag_data.index[3:7], 'high'] = [11.0, 10.9, 10.8, 10.7]
        flag_data.loc[flag_data.index[3:7], 'close'] = [10.8, 10.7, 10.6, 10.5]
        flag_data.loc[flag_data.index[3:7], 'volume'] = [7000, 6000, 5000, 4000]  # Decreasing volume
        
        # Test bull flag detection
        bull_flags = self.pattern_detector.detect_bull_flag(flag_data)
        self.assertTrue(len(bull_flags) > 0)
    
    def test_first_candle_to_make_new_high(self):
        """Test 'first candle to make a new high' pattern detection."""
        # Create a dataframe with a 'first candle to make a new high' pattern
        breakout_data = self.sample_data.copy()
        
        # Set a high, then pullback, then breakout
        breakout_data.loc[breakout_data.index[5], 'high'] = 11.0
        breakout_data.loc[breakout_data.index[6:8], 'high'] = [10.8, 10.6]  # Pullback
        breakout_data.loc[breakout_data.index[8], 'high'] = 11.1  # New high
        breakout_data.loc[breakout_data.index[8], 'open'] = 10.7
        breakout_data.loc[breakout_data.index[8], 'close'] = 11.0  # Bullish candle
        
        # Test first candle to make a new high detection
        new_highs = self.pattern_detector.detect_first_candle_to_make_new_high(breakout_data)
        self.assertTrue(len(new_highs) > 0)
    
    def test_micro_pullback_detection(self):
        """Test micro pullback pattern detection."""
        # Create a dataframe with a micro pullback pattern
        pullback_data = self.sample_data.copy()
        
        # Set an uptrend
        pullback_data.loc[pullback_data.index[5:8], 'close'] = [10.2, 10.4, 10.6]
        pullback_data.loc[pullback_data.index[5:8], 'open'] = [10.0, 10.2, 10.4]
        
        # Set a small red candle or candle with lower wick
        pullback_data.loc[pullback_data.index[8], 'open'] = 10.65
        pullback_data.loc[pullback_data.index[8], 'close'] = 10.60
        pullback_data.loc[pullback_data.index[8], 'low'] = 10.40
        pullback_data.loc[pullback_data.index[8], 'high'] = 10.70
        
        # Set a breakout candle
        pullback_data.loc[pullback_data.index[9], 'high'] = 10.90
        pullback_data.loc[pullback_data.index[9], 'close'] = 10.85
        
        # Test micro pullback detection
        pullbacks = self.pattern_detector.detect_micro_pullback(pullback_data)
        self.assertTrue(len(pullbacks) > 0)
    
    def test_edge_cases(self):
        """Test edge cases for pattern detection."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        self.assertEqual(self.pattern_detector.detect_bull_flag(empty_df), [])
        
        # Test with insufficient data
        small_df = pd.DataFrame({
            'open': [10.0, 10.2],
            'high': [10.3, 10.5],
            'low': [9.8, 10.0],
            'close': [10.2, 10.4]
        })
        self.assertEqual(self.pattern_detector.detect_bull_flag(small_df), [])
        
        # Test with index out of range
        self.assertFalse(self.pattern_detector.detect_hammer(self.sample_data, 100))


if __name__ == '__main__':
    unittest.main()