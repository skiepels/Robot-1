"""
Test Risk Manager

Unit tests for the risk management module based on Ross Cameron's strategy.
"""

import unittest
import sys
import os
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_capital = 10000.0
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
    
    def test_initialization(self):
        """Test RiskManager initialization."""
        self.assertEqual(self.risk_manager.initial_capital, self.initial_capital)
        self.assertEqual(self.risk_manager.current_capital, self.initial_capital)
        self.assertEqual(self.risk_manager.max_risk_per_trade_pct, 1.0)
        self.assertEqual(self.risk_manager.daily_max_loss_pct, 3.0)
        self.assertEqual(self.risk_manager.profit_loss_ratio, 2.0)
        self.assertEqual(self.risk_manager.max_open_positions, 3)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertEqual(self.risk_manager.total_pnl, 0.0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
        self.assertTrue(self.risk_manager.reduced_position_size)
        self.assertFalse(self.risk_manager.cushion_achieved)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with valid parameters
        entry_price = 10.0
        stop_price = 9.5
        
        # Calculate position size using 1% risk
        # For 10,000 capital, 1% risk = $100
        # Risk per share = $0.50, so expected shares = $100/$0.50 = 200
        # But with 1/4 size until quarter goal is achieved, shares = 50
        expected_size = 50
        
        actual_size = self.risk_manager.calculate_position_size(entry_price, stop_price)
        self.assertEqual(actual_size, expected_size)
        
        # Test with invalid parameters
        self.assertEqual(self.risk_manager.calculate_position_size(0, 0), 0)
        self.assertEqual(self.risk_manager.calculate_position_size(10, 11), 0)
        
    def test_validate_trade(self):
        """Test trade validation."""
        # Valid trade
        is_valid, _ = self.risk_manager.validate_trade(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=9.5,
            target_price=11.0,
            shares=100
        )
        self.assertTrue(is_valid)
        
        # Invalid: stop above entry
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=10.5,
            target_price=11.0,
            shares=100
        )
        self.assertFalse(is_valid)
        self.assertIn("Stop price must be below entry price", reason)
        
        # Invalid: poor profit-to-loss ratio
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=9.5,
            target_price=10.5,
            shares=100
        )
        self.assertFalse(is_valid)
        self.assertIn("Profit-to-loss ratio", reason)
        
        # Invalid: excessive risk
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=5.0,
            target_price=20.0,
            shares=10000
        )
        self.assertFalse(is_valid)
        self.assertIn("Trade risk", reason)
    
    def test_add_position(self):
        """Test adding a position."""
        # Add valid position
        result = self.risk_manager.add_position(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=9.5,
            target_price=11.0,
            shares=100
        )
        self.assertTrue(result)
        self.assertIn("AAPL", self.risk_manager.open_positions)
        self.assertEqual(self.risk_manager.open_positions["AAPL"]["shares"], 100)
        
        # Add invalid position
        result = self.risk_manager.add_position(
            symbol="MSFT",
            entry_price=10.0,
            stop_price=10.5,  # Invalid: stop above entry
            target_price=11.0,
            shares=100
        )
        self.assertFalse(result)
        self.assertNotIn("MSFT", self.risk_manager.open_positions)
    
    def test_close_position(self):
        """Test closing a position."""
        # Add a position
        self.risk_manager.add_position(
            symbol="AAPL",
            entry_price=10.0,
            stop_price=9.5,
            target_price=11.0,
            shares=100
        )
        
        # Close with profit
        closed_position = self.risk_manager.close_position(
            symbol="AAPL",
            exit_price=11.0,
            exit_reason="target_reached"
        )
        
        self.assertIsNotNone(closed_position)
        self.assertEqual(closed_position["exit_price"], 11.0)
        self.assertEqual(closed_position["exit_reason"], "target_reached")
        self.assertEqual(closed_position["realized_pnl"], 100.0)  # 100 shares * $1 profit
        
        # Position should be removed from open positions
        self.assertNotIn("AAPL", self.risk_manager.open_positions)
        
        # Trade history should be updated
        self.assertEqual(len(self.risk_manager.trade_history), 1)
        
        # Account metrics should be updated
        self.assertEqual(self.risk_manager.daily_pnl, 100.0)
        self.assertEqual(self.risk_manager.total_pnl, 100.0)
        self.assertEqual(self.risk_manager.current_capital, self.initial_capital + 100.0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
    
    def test_update_account_metrics(self):
        """Test updating account metrics."""
        # Update with profit
        self.risk_manager.update_account_metrics(100.0)
        
        self.assertEqual(self.risk_manager.daily_pnl, 100.0)
        self.assertEqual(self.risk_manager.total_pnl, 100.0)
        self.assertEqual(self.risk_manager.current_capital, self.initial_capital + 100.0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
        
        # Check if cushion was achieved
        quarter_goal = self.initial_capital * self.risk_manager.quarter_daily_goal
        self.assertEqual(self.risk_manager.cushion_achieved, self.risk_manager.daily_pnl >= quarter_goal)
        
        # Update with loss
        self.risk_manager.update_account_metrics(-50.0)
        
        self.assertEqual(self.risk_manager.daily_pnl, 50.0)
        self.assertEqual(self.risk_manager.total_pnl, 50.0)
        self.assertEqual(self.risk_manager.current_capital, self.initial_capital + 50.0)
        self.assertEqual(self.risk_manager.consecutive_losses, 1)
    
    def test_should_continue_trading(self):
        """Test trading continuation decision."""
        # By default, should continue with reduced size
        should_continue, reason = self.risk_manager.should_continue_trading()
        
        self.assertTrue(should_continue)
        self.assertIn("reduced position size", reason.lower())
        
        # After achieving quarter goal
        self.risk_manager.update_account_metrics(self.initial_capital * self.risk_manager.quarter_daily_goal)
        should_continue, reason = self.risk_manager.should_continue_trading()
        
        self.assertTrue(should_continue)
        self.assertIn("profit cushion", reason.lower())
        
        # After three consecutive losses
        self.risk_manager.consecutive_losses = 3
        should_continue, reason = self.risk_manager.should_continue_trading()
        
        self.assertFalse(should_continue)
        self.assertIn("consecutive losses", reason.lower())
        
        # After hitting daily loss limit
        self.risk_manager.consecutive_losses = 0
        self.risk_manager.daily_pnl = -self.initial_capital * (self.risk_manager.daily_max_loss_pct / 100)
        should_continue, reason = self.risk_manager.should_continue_trading()
        
        self.assertFalse(should_continue)
        self.assertIn("max loss", reason.lower())
    
    def test_reset_daily_metrics(self):
        """Test resetting daily metrics."""
        # First, update some metrics
        self.risk_manager.update_account_metrics(100.0)
        self.risk_manager.consecutive_losses = 2
        self.risk_manager.cushion_achieved = True
        self.risk_manager.reduced_position_size = False
        
        # Now reset
        self.risk_manager.reset_daily_metrics()
        
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertEqual(self.risk_manager.consecutive_losses, 0)
        self.assertFalse(self.risk_manager.cushion_achieved)
        self.assertTrue(self.risk_manager.reduced_position_size)


if __name__ == '__main__':
    unittest.main()