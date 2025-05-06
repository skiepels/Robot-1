"""
Test Stock Data Model

Unit tests for the Stock data model that represents stocks in the day trading strategy.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.stock import Stock


class TestStock(unittest.TestCase):
    """Test cases for the Stock class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stock = Stock('AAPL', 'Apple Inc.')
        
        # Create sample price history
        dates = pd.date_range(start='2025-05-01', periods=10, freq='1min')
        self.sample_data = pd.DataFrame({
            'open': [150.0, 150.5, 151.0, 150.8, 151.2, 151.5, 151.3, 151.8, 152.0, 152.2],
            'high': [150.8, 151.2, 151.5, 151.3, 151.8, 152.0, 151.9, 152.4, 152.6, 152.8],
            'low': [149.5, 150.1, 150.4, 150.2, 150.6, 151.0, 150.8, 151.3, 151.5, 151.7],
            'close': [150.5, 151.0, 150.8, 151.2, 151.5, 151.3, 151.8, 152.0, 152.2, 152.5],
            'volume': [10000, 12000, 9000, 11000, 15000, 13000, 14000, 16000, 18000, 20000]
        }, index=dates)
    
    def test_initialization(self):
        """Test Stock initialization."""
        self.assertEqual(self.stock.symbol, 'AAPL')
        self.assertEqual(self.stock.name, 'Apple Inc.')
        self.assertEqual(self.stock.current_price, 0.0)
        self.assertEqual(self.stock.open_price, 0.0)
        self.assertEqual(self.stock.high_price, 0.0)
        self.assertEqual(self.stock.low_price, 0.0)
        self.assertEqual(self.stock.previous_close, 0.0)
        self.assertEqual(self.stock.current_volume, 0)
        self.assertEqual(self.stock.relative_volume, 0.0)
        self.assertFalse(self.stock.has_news)
        self.assertFalse(self.stock.has_bull_flag)
        self.assertFalse(self.stock.has_micro_pullback)
        self.assertFalse(self.stock.has_new_high_breakout)
    
    def test_update_price(self):
        """Test updating price data."""
        self.stock.update_price(
            current_price=151.5,
            high_price=152.0,
            low_price=150.5,
            open_price=151.0
        )
        
        self.assertEqual(self.stock.current_price, 151.5)
        self.assertEqual(self.stock.high_price, 152.0)
        self.assertEqual(self.stock.low_price, 150.5)
        self.assertEqual(self.stock.open_price, 151.0)
        
        # Test change calculation with previous close
        self.stock.previous_close = 150.0
        self.stock.update_price(current_price=152.0)
        self.assertEqual(self.stock.change_today, 2.0)
        self.assertEqual(self.stock.change_today_percent, (2.0 / 150.0) * 100)
    
    def test_update_volume(self):
        """Test updating volume data."""
        self.stock.update_volume(current_volume=1000000, avg_volume_50d=500000)
        
        self.assertEqual(self.stock.current_volume, 1000000)
        self.assertEqual(self.stock.avg_volume_50d, 500000)
        self.assertEqual(self.stock.relative_volume, 2.0)
    
    def test_update_share_data(self):
        """Test updating share structure data."""
        self.stock.update_share_data(
            shares_outstanding=1000000000,
            shares_float=900000000,
            shares_short=50000000
        )
        
        self.assertEqual(self.stock.shares_outstanding, 1000000000)
        self.assertEqual(self.stock.shares_float, 900000000)
        self.assertEqual(self.stock.shares_short, 50000000)
        self.assertAlmostEqual(self.stock.short_ratio, 50000000 / 900000000)
    
    def test_update_technical_indicators(self):
        """Test updating technical indicators."""
        self.stock.update_technical_indicators(
            vwap=150.75,
            sma_20=149.50,
            sma_50=147.25,
            sma_200=140.00,
            ema_9=151.25
        )
        
        self.assertEqual(self.stock.vwap, 150.75)
        self.assertEqual(self.stock.sma_20, 149.50)
        self.assertEqual(self.stock.sma_50, 147.25)
        self.assertEqual(self.stock.sma_200, 140.00)
        self.assertEqual(self.stock.ema_9, 151.25)
    
    def test_add_news(self):
        """Test adding news information."""
        headline = "Apple Announces New iPhone Model"
        source = "Bloomberg"
        timestamp = datetime.now()
        
        self.stock.add_news(headline, source, timestamp)
        
        self.assertTrue(self.stock.has_news)
        self.assertEqual(self.stock.news_headline, headline)
        self.assertEqual(self.stock.news_source, source)
        self.assertEqual(self.stock.news_timestamp, timestamp)
    
    def test_set_price_history(self):
        """Test setting price history."""
        self.stock.set_price_history(self.sample_data)
        
        self.assertIsNotNone(self.stock.price_history)
        self.assertEqual(len(self.stock.price_history), 10)
        self.assertEqual(self.stock.price_history['close'].iloc[-1], 152.5)
    
    def test_meets_criteria(self):
        """Test criteria validation."""
        # Set values that should pass criteria
        self.stock.current_price = 15.0
        self.stock.gap_percent = 12.0
        self.stock.relative_volume = 6.0
        self.stock.shares_float = 8_000_000
        
        # Test with default criteria
        self.assertTrue(self.stock.meets_criteria())
        
        # Test with failing criteria
        self.stock.current_price = 25.0  # Outside price range
        self.assertFalse(self.stock.meets_criteria())
        
        self.stock.current_price = 15.0
        self.stock.gap_percent = 8.0  # Below minimum gap
        self.assertFalse(self.stock.meets_criteria())
        
        self.stock.gap_percent = 12.0
        self.stock.relative_volume = 4.0  # Below minimum relative volume
        self.assertFalse(self.stock.meets_criteria())
        
        self.stock.relative_volume = 6.0
        self.stock.shares_float = 12_000_000  # Above maximum float
        self.assertFalse(self.stock.meets_criteria())
    
    def test_get_risk_level(self):
        """Test risk level calculation."""
        # Low price, high relative volume, low float = high risk
        self.stock.current_price = 2.0
        self.stock.relative_volume = 10.0
        self.stock.shares_float = 2_000_000
        
        risk_level = self.stock.get_risk_level()
        self.assertGreater(risk_level, 7.0)  # Should be high risk
        
        # Higher price, lower relative volume, higher float = lower risk
        self.stock.current_price = 18.0
        self.stock.relative_volume = 3.0
        self.stock.shares_float = 8_000_000
        
        risk_level = self.stock.get_risk_level()
        self.assertLess(risk_level, 5.0)  # Should be medium/low risk
    
    def test_get_optimal_entry_stop_target(self):
        """Test optimal entry, stop loss, and target calculations."""
        # Set up a stock with price history and pattern flags
        self.stock.set_price_history(self.sample_data)
        
        # Test bull flag pattern
        self.stock.has_bull_flag = True
        self.stock.has_micro_pullback = False
        self.stock.has_new_high_breakout = False
        
        entry = self.stock.get_optimal_entry()
        stop = self.stock.get_optimal_stop_loss()
        target = self.stock.get_optimal_target()
        
        self.assertIsNotNone(entry)
        self.assertIsNotNone(stop)
        self.assertIsNotNone(target)
        self.assertGreater(entry, stop)  # Entry should be above stop
        self.assertGreater(target, entry)  # Target should be above entry
        
        # Test with no pattern
        self.stock.has_bull_flag = False
        self.stock.has_micro_pullback = False
        self.stock.has_new_high_breakout = False
        
        entry = self.stock.get_optimal_entry()
        self.assertIsNotNone(entry)  # Should default to current price
    
    def test_to_string(self):
        """Test string representation."""
        self.stock.current_price = 150.50
        self.stock.change_today_percent = 2.5
        self.stock.current_volume = 5000000
        self.stock.relative_volume = 1.5
        self.stock.shares_float = 900000000
        
        string_repr = str(self.stock)
        
        self.assertIn("AAPL", string_repr)
        self.assertIn("$150.50", string_repr)
        self.assertIn("2.50%", string_repr)
        self.assertIn("5,000,000", string_repr)
        self.assertIn("1.50x", string_repr)
        self.assertIn("900,000,000", string_repr)


if __name__ == '__main__':
    unittest.main()