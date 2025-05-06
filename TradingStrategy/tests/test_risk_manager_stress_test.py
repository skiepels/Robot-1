"""
Risk Manager Stress Test

This script performs stress testing on the risk management system to ensure it behaves
properly under various conditions and edge cases.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.risk_manager import RiskManager


class RiskManagerStressTest:
    """Stress test for the RiskManager class."""
    
    def __init__(self, initial_capital=10000.0):
        """Initialize the stress test with initial capital."""
        self.initial_capital = initial_capital
        
        # Create output directory
        os.makedirs('logs/risk_tests', exist_ok=True)
    
    def test_position_sizing(self):
        """Test position sizing under various conditions."""
        print("\n=== Testing Position Sizing ===")
        
        # Create risk manager with default parameters
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Test cases for position sizing
        test_cases = [
            # Entry, Stop, Expected Shares with Quarter Size (initial state)
            (10.0, 9.5, 50),      # $0.50 risk per share, $100 max risk (1%), quarter size = 50 shares
            (50.0, 48.0, 10),      # $2.00 risk per share, quarter size = 12 shares
            (5.0, 4.8, 104),       # $0.20 risk per share, quarter size = 125 shares
            (100.0, 99.0, 10),     # $1.00 risk per share, quarter size = 25 shares
            (20.0, 19.0, 25),      # $1.00 risk per share, quarter size = 25 shares
        ]
        
        # Test position sizing with quarter size (default at start)
        print("\nPosition Sizing with Quarter Size (Initial):")
        print("Entry Price\tStop Price\tRisk Per Share\tExpected Shares\tActual Shares")
        
        for entry, stop, expected in test_cases:
            risk_per_share = entry - stop
            actual = risk_manager.calculate_position_size(entry, stop)
            print(f"${entry:.2f}\t${stop:.2f}\t${risk_per_share:.2f}\t\t{expected}\t\t{actual}")
            
            # Verify calculation
            if actual != expected:
                print(f"  WARNING: Expected {expected} shares, got {actual}")
        
        # Now test with full size after profit cushion
        # Set profit cushion
        quarter_goal = risk_manager.initial_capital * risk_manager.quarter_daily_goal
        risk_manager.update_account_metrics(quarter_goal + 10.0)  # Add buffer
        
        # Double the expected shares for full size
        full_size_cases = [(entry, stop, expected * 4) for entry, stop, expected in test_cases]
        
        print("\nPosition Sizing with Full Size (After Profit Cushion):")
        print("Entry Price\tStop Price\tRisk Per Share\tExpected Shares\tActual Shares")
        
        for entry, stop, expected in full_size_cases:
            risk_per_share = entry - stop
            actual = risk_manager.calculate_position_size(entry, stop)
            print(f"${entry:.2f}\t${stop:.2f}\t${risk_per_share:.2f}\t\t{expected}\t\t{actual}")
            
            # Verify calculation
            if actual != expected:
                print(f"  WARNING: Expected {expected} shares, got {actual}")
        
        # Test with various capital levels
        capital_levels = [1000.0, 5000.0, 25000.0, 100000.0]
        
        print("\nPosition Sizing with Different Capital Levels:")
        print("Capital\t\tEntry\tStop\tRisk/Share\tShares")
        
        for capital in capital_levels:
            test_risk_manager = RiskManager(
                initial_capital=capital,
                max_risk_per_trade_pct=1.0
            )
            
            entry = 10.0
            stop = 9.5
            risk_per_share = entry - stop
            max_risk = capital * 0.01  # 1%
            
            # Expected shares with quarter size
            expected = int((max_risk / 4) / risk_per_share)
            actual = test_risk_manager.calculate_position_size(entry, stop)
            
            print(f"${capital:.2f}\t${entry:.2f}\t${stop:.2f}\t${risk_per_share:.2f}\t\t{actual}")
        
        # Plot position size vs. risk per share
        risk_per_share_values = np.linspace(0.05, 2.0, 40)  # From $0.05 to $2.00
        position_sizes = []
        
        test_risk_manager = RiskManager(
            initial_capital=10000.0,
            max_risk_per_trade_pct=1.0
        )
        
        entry_price = 10.0
        for risk in risk_per_share_values:
            stop_price = entry_price - risk
            size = test_risk_manager.calculate_position_size(entry_price, stop_price)
            position_sizes.append(size)
        
        plt.figure(figsize=(10, 6))
        plt.plot(risk_per_share_values, position_sizes, marker='o')
        plt.title('Position Size vs. Risk Per Share')
        plt.xlabel('Risk Per Share ($)')
        plt.ylabel('Position Size (Shares)')
        plt.grid(True)
        plt.savefig('logs/risk_tests/position_size_vs_risk.png')
        
        print(f"\nPosition size vs. risk chart saved to logs/risk_tests/position_size_vs_risk.png")
    
    def test_trade_validation(self):
        """Test trade validation under various conditions."""
        print("\n=== Testing Trade Validation ===")
        
        # Create risk manager with default parameters
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Test cases for trade validation
        test_cases = [
            # Symbol, Entry, Stop, Target, Shares, Expected Valid, Reason
            ("AAPL", 10.0, 9.5, 11.0, 100, True, "Valid trade"),
            ("MSFT", 10.0, 10.5, 11.0, 100, False, "Stop above entry"),
            ("GOOGL", 10.0, 9.5, 10.5, 100, False, "Poor profit-loss ratio"),
            ("AMZN", 10.0, 9.0, 12.0, 1000, False, "Excessive risk"),
        ]
        
        print("\nTrade Validation Tests:")
        print("Symbol\tEntry\tStop\tTarget\tShares\tValid?\tReason")
        
        for symbol, entry, stop, target, shares, expected_valid, reason in test_cases:
            is_valid, validation_reason = risk_manager.validate_trade(
                symbol, entry, stop, target, shares
            )
            
            print(f"{symbol}\t${entry:.2f}\t${stop:.2f}\t${target:.2f}\t{shares}\t{is_valid}\t{validation_reason}")
            
            # Verify validation
            if is_valid != expected_valid:
                print(f"  WARNING: Expected valid={expected_valid}, got valid={is_valid}")
        
        # Test max positions validation
        print("\nTesting Maximum Positions Limit:")
        
        # Add max positions
        for i in range(risk_manager.max_open_positions):
            symbol = f"STOCK{i+1}"
            added = risk_manager.add_position(
                symbol=symbol,
                entry_price=10.0,
                stop_price=9.5,
                target_price=11.0,
                shares=100
            )
            print(f"Added position {i+1}: {symbol} - Success: {added}")
        
        # Try to add one more
        added = risk_manager.add_position(
            symbol="EXTRA",
            entry_price=10.0,
            stop_price=9.5,
            target_price=11.0,
            shares=100
        )
        print(f"Attempted to add extra position: Success: {added}")
        
        # Validate another trade
        is_valid, reason = risk_manager.validate_trade(
            "EXTRA", 10.0, 9.5, 11.0, 100
        )
        print(f"Validate extra trade: Valid: {is_valid}, Reason: {reason}")
    
    def test_account_metrics(self):
        """Test account metrics and tracking."""
        print("\n=== Testing Account Metrics ===")
        
        # Create risk manager with default parameters
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        initial_capital = risk_manager.current_capital
        
        # Simulate a series of trades and track metrics
        trades = [
            {"pnl": 100.0, "win": True},
            {"pnl": 75.0, "win": True},
            {"pnl": -50.0, "win": False},
            {"pnl": 120.0, "win": True},
            {"pnl": -40.0, "win": False},
            {"pnl": -45.0, "win": False},
            {"pnl": 90.0, "win": True},
            {"pnl": 60.0, "win": True},
        ]
        
        # Track metrics for each trade
        equity_curve = [initial_capital]
        daily_pnl = []
        win_rates = []
        metrics = []
        
        print("\nTrade Sequence and Impact on Metrics:")
        print("Trade\tP&L\tEquity\tDaily P&L\tWin Rate\tConsec. Losses\tCushion")
        
        for i, trade in enumerate(trades):
            risk_manager.update_account_metrics(trade["pnl"], trade["win"])
            
            equity_curve.append(risk_manager.current_capital)
            daily_pnl.append(risk_manager.daily_pnl)
            
            # Calculate win rate
            win_rate = 0
            if i > 0:
                wins = sum(1 for t in trades[:i+1] if t["win"])
                win_rate = wins / (i + 1)
            win_rates.append(win_rate)
            
            # Save for plotting
            metrics.append({
                "trade": i + 1,
                "pnl": trade["pnl"],
                "equity": risk_manager.current_capital,
                "daily_pnl": risk_manager.daily_pnl,
                "win_rate": win_rate,
                "consecutive_losses": risk_manager.consecutive_losses,
                "cushion_achieved": risk_manager.cushion_achieved
            })
            
            print(f"{i+1}\t${trade['pnl']:.2f}\t${risk_manager.current_capital:.2f}\t${risk_manager.daily_pnl:.2f}\t{win_rate:.2f}\t{risk_manager.consecutive_losses}\t{risk_manager.cushion_achieved}")
        
        # Plot equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(equity_curve)), equity_curve, marker='o')
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Account Equity ($)')
        plt.grid(True)
        plt.savefig('logs/risk_tests/equity_curve.png')
        
        # Plot daily P&L
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(daily_pnl) + 1), daily_pnl)
        plt.title('Daily P&L')
        plt.xlabel('Trade Number')
        plt.ylabel('P&L ($)')
        plt.grid(True)
        plt.savefig('logs/risk_tests/daily_pnl.png')
        
        # Plot win rate progression
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(win_rates) + 1), win_rates, marker='o')
        plt.title('Win Rate Progression')
        plt.xlabel('Trade Number')
        plt.ylabel('Win Rate')
        plt.grid(True)
        plt.savefig('logs/risk_tests/win_rate.png')
        
        print("\nCharts saved to logs/risk_tests/ directory")
    
    def test_consecutive_losses(self):
        """Test the system's response to consecutive losing trades."""
        print("\n=== Testing Consecutive Losses Handling ===")
        
        # Create risk manager with default parameters
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Test consecutive losses tracking
        print("\nConsecutive Losses Tracking:")
        print("Trade\tP&L\tConsecutive Losses\tShould Continue Trading?")
        
        # Start with a winning trade
        risk_manager.update_account_metrics(100.0, True)
        should_continue, reason = risk_manager.should_continue_trading()
        print(f"1\t$100.00\t{risk_manager.consecutive_losses}\t\t{should_continue}")
        
        # Add consecutive losing trades
        for i in range(3):
            risk_manager.update_account_metrics(-50.0, False)
            should_continue, reason = risk_manager.should_continue_trading()
            print(f"{i+2}\t-$50.00\t{risk_manager.consecutive_losses}\t\t{should_continue} - {reason}")
        
        # Verify that after 3 consecutive losses, trading should stop
        if should_continue:
            print("  WARNING: Should have stopped trading after 3 consecutive losses")
        
        # Reset and test daily loss limit
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        print("\nDaily Loss Limit Testing:")
        print("Trade\tP&L\tDaily P&L\tShould Continue Trading?")
        
        # Add losing trades until hitting daily loss limit (3% of capital)
        daily_loss_limit = risk_manager.initial_capital * (risk_manager.daily_max_loss_pct / 100)
        remaining_loss = daily_loss_limit
        trade_count = 0
        
        while remaining_loss > 0:
            trade_count += 1
            loss_amount = min(100.0, remaining_loss)
            risk_manager.update_account_metrics(-loss_amount, False)
            remaining_loss -= loss_amount
            
            should_continue, reason = risk_manager.should_continue_trading()
            print(f"{trade_count}\t-${loss_amount:.2f}\t-${risk_manager.daily_pnl:.2f}\t{should_continue} - {reason}")
        
        # Verify that after hitting daily loss limit, trading should stop
        if should_continue:
            print("  WARNING: Should have stopped trading after hitting daily loss limit")
    
    def test_position_adjustment(self):
        """Test position adjustment logic (trailing stops, adding to winners)."""
        print("\n=== Testing Position Adjustment ===")
        
        # Create risk manager with default parameters
        risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Add a position
        symbol = "AAPL"
        entry_price = 150.0
        stop_price = 145.0
        target_price = 160.0
        shares = 20
        
        risk_manager.add_position(symbol, entry_price, stop_price, target_price, shares)
        
        # Check for adjustments at various price levels
        price_levels = [
            151.0,  # Small profit, no adjustment expected
            153.0,  # More profit, still no adjustment
            155.0,  # Significant profit, trailing stop adjustment expected
            158.0,  # Near target, trailing stop adjustment expected
        ]
        
        print("\nTrailing Stop Adjustments:")
        print("Current Price\tUnrealized P&L\tOriginal Stop\tNew Stop\tReason")
        
        for current_price in price_levels:
            # Update position with current price
            position = risk_manager.update_position(symbol, current_price)
            
            # Check for adjustments
            adjustments = risk_manager.check_for_adjustments()
            
            if adjustments:
                for adj in adjustments:
                    if adj['action'] == 'adjust_stop':
                        print(f"${current_price:.2f}\t${position['unrealized_pnl']:.2f}\t\t${adj['current_stop']:.2f}\t\t${adj['new_stop']:.2f}\t{adj['reason']}")
                        
                        # Update stop price
                        risk_manager.open_positions[symbol]['stop_price'] = adj['new_stop']
            else:
                print(f"${current_price:.2f}\t${position['unrealized_pnl']:.2f}\t\t${position['stop_price']:.2f}\t\t-\t\tNo adjustment")
        
        # Test adding to winners
        # First, establish profit cushion
        quarter_goal = risk_manager.initial_capital * risk_manager.quarter_daily_goal
        risk_manager.update_account_metrics(quarter_goal + 10.0)  # Add buffer
        
        print("\nAdding to Winners:")
        print("Current Price\tUnrealized P&L\tCurrent Shares\tAdditional Shares\tReason")
        
        # Now check for adding to winner at a strong profit level
        current_price = 159.0
        position = risk_manager.update_position(symbol, current_price)
        
        # Check for adjustments
        adjustments = risk_manager.check_for_adjustments()
        
        if adjustments:
            for adj in adjustments:
                if adj['action'] == 'add_shares':
                    print(f"${current_price:.2f}\t${position['unrealized_pnl']:.2f}\t\t{adj['current_shares']}\t\t{adj['additional_shares']}\t\t{adj['reason']}")
        else:
            print(f"${current_price:.2f}\t${position['unrealized_pnl']:.2f}\t\t{position['shares']}\t\t-\t\tNo additional shares")
    
    def run_all_tests(self):
        """Run all stress tests."""
        self.test_position_sizing()
        self.test_trade_validation()
        self.test_account_metrics()
        self.test_consecutive_losses()
        self.test_position_adjustment()


if __name__ == '__main__':
    stress_test = RiskManagerStressTest(initial_capital=10000.0)
    stress_test.run_all_tests()