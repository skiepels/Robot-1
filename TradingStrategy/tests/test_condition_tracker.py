"""
Test Condition Tracker

Unit tests for the ConditionTracker class that monitors market conditions and trading patterns.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scanning.condition_tracker import ConditionTracker
from src.data.stock import Stock


class TestConditionTracker(unittest.TestCase):
    """Test cases for the ConditionTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock instances for dependencies
        self.mock_market_data = MagicMock()
        self.mock_news_provider = MagicMock()
        
        # Create the condition tracker
        self.condition_tracker = ConditionTracker(self.mock_market_data, self.mock_news_provider)
        
        # Setup sample stock data
        self.sample_stocks = [
            Stock('AAPL', 'Apple Inc.'),
            Stock('MSFT', 'Microsoft Corp.'),
            Stock('TSLA', 'Tesla Inc.')
        ]
        
        # Setup stock attributes to meet basic criteria
        for stock in self.sample_stocks:
            stock.current_price = 15.0
            stock.gap_percent = 12.0
            stock.relative_volume = 6.0
            stock.shares_float = 8_000_000
            stock.has_news = False
        
        # Sample OHLCV data for pattern detection
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
                     7500, 5000, 6000, 8000, 5500, 6500, 9000, 6000, 7000, 10000],
            'vwap': [10.1, 10.3, 10.3, 10.2, 10.4, 10.4, 10.3, 10.5, 10.6, 10.4,
                    10.6, 10.7, 10.5, 10.7, 10.8, 10.6, 10.8, 10.9, 10.7, 10.9]
        }, index=dates)
    
    def test_initialization(self):
        """Test ConditionTracker initialization."""
        self.assertEqual(self.condition_tracker.market_data, self.mock_market_data)
        self.assertEqual(self.condition_tracker.news, self.mock_news_provider)
        self.assertFalse(self.condition_tracker.market_open)
        self.assertFalse(self.condition_tracker.pre_market)
        self.assertFalse(self.condition_tracker.post_market)
        self.assertFalse(self.condition_tracker.strong_market)
        self.assertEqual(len(self.condition_tracker.tracked_stocks), 0)
        self.assertEqual(len(self.condition_tracker.bull_flags), 0)
        self.assertEqual(len(self.condition_tracker.micro_pullbacks), 0)
        self.assertEqual(len(self.condition_tracker.new_high_breakouts), 0)
    
    def test_update_market_session(self):
        """Test updating market session status."""
        # Mock datetime.now() to return a specific time
        # Regular market hours (9:30 AM - 4:00 PM Eastern)
        with patch('src.scanning.condition_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.strptime("2025-05-05 11:30:00", "%Y-%m-%d %H:%M:%S")
            mock_datetime.strptime = datetime.strptime
            
            self.condition_tracker.update_market_session()
            
            self.assertTrue(self.condition_tracker.market_open)
            self.assertFalse(self.condition_tracker.pre_market)
            self.assertFalse(self.condition_tracker.post_market)
        
        # Pre-market hours (4:00 AM - 9:30 AM Eastern)
        with patch('src.scanning.condition_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.strptime("2025-05-05 08:30:00", "%Y-%m-%d %H:%M:%S")
            mock_datetime.strptime = datetime.strptime
            
            self.condition_tracker.update_market_session()
            
            self.assertFalse(self.condition_tracker.market_open)
            self.assertTrue(self.condition_tracker.pre_market)
            self.assertFalse(self.condition_tracker.post_market)
        
        # After-hours (4:00 PM - 8:00 PM Eastern)
        with patch('src.scanning.condition_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.strptime("2025-05-05 17:30:00", "%Y-%m-%d %H:%M:%S")
            mock_datetime.strptime = datetime.strptime
            
            self.condition_tracker.update_market_session()
            
            self.assertFalse(self.condition_tracker.market_open)
            self.assertFalse(self.condition_tracker.pre_market)
            self.assertTrue(self.condition_tracker.post_market)
    
    def test_update_market_conditions(self):
        """Test updating market conditions."""
        # Mock data for indices
        spy_data = self.sample_data.copy()
        spy_data['close'] = [item * 1.05 for item in spy_data['close']]  # Uptrend
        
        qqq_data = self.sample_data.copy()
        qqq_data['close'] = [item * 1.07 for item in qqq_data['close']]  # Strong uptrend
        
        iwm_data = self.sample_data.copy()
        iwm_data['close'] = [item * 0.98 for item in iwm_data['close']]  # Downtrend
        
        # Mock get_intraday_data to return our sample data
        self.mock_market_data.get_intraday_data.side_effect = lambda symbol, **kwargs: {
            'SPY': spy_data,
            'QQQ': qqq_data,
            'IWM': iwm_data
        }.get(symbol, pd.DataFrame())
        
        # Update market conditions
        self.condition_tracker.update_market_conditions()
        
        # 2 out of 3 indices are up, so market should be strong
        self.assertTrue(self.condition_tracker.strong_market)
        
        # Test with all indices down
        spy_data['close'] = [item * 0.95 for item in spy_data['close']]
        qqq_data['close'] = [item * 0.93 for item in qqq_data['close']]
        
        # Update market conditions again
        self.condition_tracker.update_market_conditions()
        
        # All indices down, market should not be strong
        self.assertFalse(self.condition_tracker.strong_market)
    
    def test_scan_for_trading_conditions(self):
        """Test scanning for trading conditions."""
        # Set up news items for each stock
        self.mock_news_provider.get_stock_news.return_value = [
            MagicMock(headline="Test News", source="Test Source", date=datetime.now())
        ]
        
        # Set up sample data for each stock
        bull_flag_data = self.sample_data.copy()
        # Modify data to show bull flag pattern
        bull_flag_data.iloc[0:3, bull_flag_data.columns.get_indexer(['close'])] = [10.0, 10.5, 11.0]
        bull_flag_data.iloc[3:7, bull_flag_data.columns.get_indexer(['high'])] = [11.0, 10.9, 10.8, 10.7]
        
        # Mock get_intraday_data to return bull flag pattern for AAPL
        self.mock_market_data.get_intraday_data.side_effect = lambda symbol, **kwargs: {
            'AAPL': bull_flag_data,
            'MSFT': self.sample_data,
            'TSLA': self.sample_data
        }.get(symbol, pd.DataFrame())
        
        # Scan for trading conditions
        tracked_stocks = self.condition_tracker.scan_for_trading_conditions(
            self.sample_stocks, min_gap_pct=10.0, min_rel_volume=5.0, max_float=10_000_000
        )
        
        # Should find bull flag on AAPL
        self.assertIn('AAPL', tracked_stocks)
        self.assertEqual(tracked_stocks['AAPL'], 'bull_flag')
        self.assertEqual(len(self.condition_tracker.bull_flags), 1)
    
    def test_get_actionable_stocks(self):
        """Test getting actionable stocks."""
        # Setup mock data
        # AAPL with bull flag
        stock1 = Stock('AAPL', 'Apple Inc.')
        stock1.has_bull_flag = True
        self.condition_tracker.bull_flags['AAPL'] = stock1
        self.condition_tracker.tracked_stocks['AAPL'] = 'bull_flag'
        
        # MSFT with new high breakout
        stock2 = Stock('MSFT', 'Microsoft Corp.')
        stock2.has_new_high_breakout = True
        self.condition_tracker.new_high_breakouts['MSFT'] = stock2
        self.condition_tracker.tracked_stocks['MSFT'] = 'new_high_breakout'
        
        # TSLA with micro pullback
        stock3 = Stock('TSLA', 'Tesla Inc.')
        stock3.has_micro_pullback = True
        self.condition_tracker.micro_pullbacks['TSLA'] = stock3
        self.condition_tracker.tracked_stocks['TSLA'] = 'micro_pullback'
        
        # Get actionable stocks
        actionable_stocks = self.condition_tracker.get_actionable_stocks(max_stocks=2)
        
        # Should prioritize bull flag and new high breakout
        self.assertEqual(len(actionable_stocks), 2)
        self.assertEqual(actionable_stocks[0].symbol, 'AAPL')  # Bull flag first
        self.assertEqual(actionable_stocks[1].symbol, 'MSFT')  # New high breakout second
    
    def test_generate_alerts(self):
        """Test generating trading alerts."""
        # Setup mock data for stocks with patterns
        stock1 = Stock('AAPL', 'Apple Inc.')
        stock1.has_bull_flag = True
        stock1.current_price = 150.0
        stock1.set_price_history(self.sample_data.copy())
        self.condition_tracker.bull_flags['AAPL'] = stock1
        self.condition_tracker.tracked_stocks['AAPL'] = 'bull_flag'
        
        # Get alerts
        alerts = self.condition_tracker.generate_alerts()
        
        # Should have an alert for AAPL
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['symbol'], 'AAPL')
        self.assertEqual(alerts[0]['pattern'], 'bull_flag')
        self.assertIn('entry', alerts[0])
        self.assertIn('stop_loss', alerts[0])
        self.assertIn('target', alerts[0])
        self.assertIn('message', alerts[0])
    
    def test_is_market_healthy(self):
        """Test market health check."""
        # Setup mock for update_market_conditions
        self.condition_tracker.update_market_conditions = MagicMock()
        self.condition_tracker.update_market_session = MagicMock()
        
        # Test with strong market during market hours
        self.condition_tracker.strong_market = True
        self.condition_tracker.market_open = True
        self.assertTrue(self.condition_tracker.is_market_healthy())
        
        # Test with weak market during market hours
        self.condition_tracker.strong_market = False
        self.condition_tracker.market_open = True
        self.assertFalse(self.condition_tracker.is_market_healthy())
        
        # Test with strong market during pre-market
        self.condition_tracker.strong_market = True
        self.condition_tracker.market_open = False
        self.condition_tracker.pre_market = True
        self.assertTrue(self.condition_tracker.is_market_healthy())
        
        # Test with strong market after hours (should be unhealthy)
        self.condition_tracker.strong_market = True
        self.condition_tracker.market_open = False
        self.condition_tracker.pre_market = False
        self.condition_tracker.post_market = True
        self.assertFalse(self.condition_tracker.is_market_healthy())
    
    def test_should_use_reduced_size(self):
        """Test position sizing decision."""
        # Test below quarter goal
        current_profit = 25.0
        daily_goal = 200.0
        self.assertTrue(self.condition_tracker.should_use_reduced_size(current_profit, daily_goal))
        
        # Test above quarter goal
        current_profit = 60.0
        daily_goal = 200.0
        self.assertFalse(self.condition_tracker.should_use_reduced_size(current_profit, daily_goal))


if __name__ == '__main__':
    unittest.main()