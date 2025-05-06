"""
Risk Manager

This module implements risk management strategies based on Ross Cameron's approach,
including position sizing, profit targets, stop losses, and overall portfolio risk management.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, initial_capital, max_risk_per_trade_pct=1.0, daily_max_loss_pct=3.0, 
                profit_loss_ratio=2.0, max_open_positions=3):
        """
        Initialize the risk manager.
        
        Parameters:
        -----------
        initial_capital: float
            Starting capital for the trading account
        max_risk_per_trade_pct: float
            Maximum percentage of account to risk on any single trade
        daily_max_loss_pct: float
            Maximum percentage of account to lose in a single day
        profit_loss_ratio: float
            Target ratio of profit to loss (e.g., 2.0 means targeting $2 profit for every $1 risked)
        max_open_positions: int
            Maximum number of open positions allowed at once
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.daily_max_loss_pct = daily_max_loss_pct
        self.profit_loss_ratio = profit_loss_ratio
        self.max_open_positions = max_open_positions
        
        # Track daily P&L
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Track trade performance
        self.trade_history = []
        self.open_positions = {}
        self.consecutive_losses = 0
        
        # Ross Cameron's strategy for small accounts
        self.quarter_daily_goal = 0.005  # 0.5% of account as quarter daily goal
        self.daily_goal = 0.02  # 2% of account as daily goal
        self.cushion_achieved = False
        self.reduced_position_size = True  # Start with reduced size
        
        # Additional risk metrics
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
    
    def calculate_position_size(self, entry_price, stop_price, stock=None):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Parameters:
        -----------
        entry_price: float
            Planned entry price for the trade
        stop_price: float
            Planned stop loss price for the trade
        stock: Stock, optional
            Stock object for additional risk assessment
            
        Returns:
        --------
        int: Number of shares to trade
        """
        if entry_price <= 0 or stop_price <= 0 or entry_price <= stop_price:
            logger.warning("Invalid prices for position sizing calculation")
            return 0
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_price
        
        # Calculate dollar risk allowance based on account size
        dollar_risk = self.current_capital * (self.max_risk_per_trade_pct / 100)
        
        # Apply Ross Cameron's small account strategy
        if self.reduced_position_size:
            # Use quarter size until reaching the quarter daily goal
            dollar_risk = dollar_risk / 4
        
        # Calculate share size
        if risk_per_share > 0:
            shares = int(dollar_risk / risk_per_share)
        else:
            shares = 0
        
        # Apply additional constraints
        if stock:
            # Adjust for stock-specific risk
            risk_level = stock.get_risk_level() if hasattr(stock, 'get_risk_level') else 5
            risk_multiplier = max(1.0 - (risk_level / 20.0), 0.5)
            shares = int(shares * risk_multiplier)
        
        # Additional constraints based on current market conditions
        if not self.cushion_achieved and self.daily_pnl <= 0:
            # If no profit cushion and day is not green, limit size further
            shares = int(shares * 0.75)
        
        # Enforce minimum share size
        if shares < 1:
            shares = 0  # Skip trade if too small
            
        logger.info(f"Calculated position size: {shares} shares at ${entry_price:.2f} with stop at ${stop_price:.2f}")
        
        return shares
    
    def update_account_metrics(self, trade_pnl, is_win=None):
        """
        Update account metrics after a trade.
        
        Parameters:
        -----------
        trade_pnl: float
            Profit/loss from the trade
        is_win: bool, optional
            Explicitly specify if trade was a win
        """
        # Update daily P&L
        self.daily_pnl += trade_pnl
        self.total_pnl += trade_pnl
        
        # Update current capital
        self.current_capital += trade_pnl
        
        # Update drawdown
        if trade_pnl < 0:
            self.drawdown += trade_pnl / self.initial_capital * 100.0
            self.max_drawdown = min(self.max_drawdown, self.drawdown)
        else:
            self.drawdown = 0.0  # Reset drawdown after a profitable trade
        
        # Update win/loss streak
        if is_win is None:
            is_win = trade_pnl > 0
            
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Update quarter goal achievement status
        quarter_goal_amount = self.current_capital * self.quarter_daily_goal
        self.cushion_achieved = self.daily_pnl >= quarter_goal_amount
        
        # Update position sizing strategy
        self.reduced_position_size = not self.cushion_achieved
        
        # Update performance metrics
        self._update_performance_metrics(trade_pnl, is_win)
        
        logger.info(f"Account metrics updated - Daily P&L: ${self.daily_pnl:.2f}, " +
                   f"Cushion achieved: {self.cushion_achieved}, " +
                   f"Consecutive losses: {self.consecutive_losses}")
    
    def validate_trade(self, symbol, entry_price, stop_price, target_price, shares):
        """
        Validate a potential trade against risk management rules.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        entry_price: float
            Planned entry price
        stop_price: float
            Planned stop loss price
        target_price: float
            Planned profit target price
        shares: int
            Number of shares
            
        Returns:
        --------
        tuple: (is_valid, reason_if_invalid)
        """
        # Check if price inputs are valid
        if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
            return False, "Invalid price inputs"
        
        # Check if shares is valid
        if shares <= 0:
            return False, "Invalid share quantity"
        
        # Calculate risk and reward
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        
        # Check if stop is below entry for long trades
        if risk_per_share <= 0:
            return False, "Stop price must be below entry price for long trades"
        
        # Check profit-to-loss ratio
        current_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        if current_ratio < self.profit_loss_ratio:
            return False, f"Profit-to-loss ratio ({current_ratio:.2f}) below minimum ({self.profit_loss_ratio:.2f})"
        
        # Calculate total dollar risk
        total_risk = risk_per_share * shares
        max_allowed_risk = self.current_capital * (self.max_risk_per_trade_pct / 100)
        
        if total_risk > max_allowed_risk:
            return False, f"Trade risk (${total_risk:.2f}) exceeds maximum allowed (${max_allowed_risk:.2f})"
        
        # Check daily loss limit
        daily_max_loss = self.current_capital * (self.daily_max_loss_pct / 100)
        if self.daily_pnl - total_risk < -daily_max_loss:
            return False, f"Potential loss would exceed daily max loss (${daily_max_loss:.2f})"
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            return False, f"Too many consecutive losses ({self.consecutive_losses})"
        
        # Check number of open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Maximum open positions ({self.max_open_positions}) reached"
        
        # All checks passed
        return True, "Trade validated"
    
    def add_position(self, symbol, entry_price, stop_price, target_price, shares):
        """
        Add a new position to the portfolio.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        entry_price: float
            Entry price
        stop_price: float
            Stop loss price
        target_price: float
            Profit target price
        shares: int
            Number of shares
            
        Returns:
        --------
        bool: True if position was added successfully
        """
        # Validate the trade first
        is_valid, reason = self.validate_trade(symbol, entry_price, stop_price, target_price, shares)
        
        if not is_valid:
            logger.warning(f"Trade validation failed: {reason}")
            return False
        
        # Add position
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'entry_time': datetime.now(),
            'risk_per_share': entry_price - stop_price,
            'profit_target': target_price - entry_price,
            'dollar_risk': (entry_price - stop_price) * shares,
            'status': 'open'
        }
        
        self.open_positions[symbol] = position
        
        logger.info(f"Position added: {symbol} - {shares} shares at ${entry_price:.2f}, " +
                   f"Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
        
        return True
    
    def update_position(self, symbol, current_price):
        """
        Update an existing position with current price data.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        current_price: float
            Current market price
            
        Returns:
        --------
        dict: Updated position data
        """
        if symbol not in self.open_positions:
            logger.warning(f"Position not found: {symbol}")
            return None
        
        position = self.open_positions[symbol]
        position['current_price'] = current_price
        
        # Calculate unrealized P&L
        entry_price = position['entry_price']
        shares = position['shares']
        unrealized_pnl = (current_price - entry_price) * shares
        position['unrealized_pnl'] = unrealized_pnl
        
        # Check for stop loss or target price hit
        if current_price <= position['stop_price']:
            position['status'] = 'stop_triggered'
        elif current_price >= position['target_price']:
            position['status'] = 'target_reached'
        
        return position
    
    def close_position(self, symbol, exit_price=None, exit_reason='manual'):
        """
        Close an existing position.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        exit_price: float, optional
            Exit price (uses current price if not specified)
        exit_reason: str, optional
            Reason for closing the position
            
        Returns:
        --------
        dict: Closed position data
        """
        if symbol not in self.open_positions:
            logger.warning(f"Position not found: {symbol}")
            return None
        
        position = self.open_positions[symbol]
        
        # Use provided exit price or current price
        if exit_price is None:
            exit_price = position['current_price']
        
        # Calculate realized P&L
        entry_price = position['entry_price']
        shares = position['shares']
        realized_pnl = (exit_price - entry_price) * shares
        
        # Update position data
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['realized_pnl'] = realized_pnl
        position['status'] = 'closed'
        
        # Calculate duration
        duration = position['exit_time'] - position['entry_time']
        position['duration_seconds'] = duration.total_seconds()
        
        # Update account metrics
        is_win = realized_pnl > 0
        self.update_account_metrics(realized_pnl, is_win)
        
        # Add to trade history
        self.trade_history.append(position)
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        logger.info(f"Position closed: {symbol} - {shares} shares at ${exit_price:.2f}, " +
                   f"P&L: ${realized_pnl:.2f}, Reason: {exit_reason}")
        
        return position
    
    def check_for_adjustments(self):
        """
        Check if any position adjustments are needed based on current market conditions.
        
        Returns:
        --------
        list: Recommended position adjustments
        """
        adjustments = []
        
        # Check each open position
        for symbol, position in self.open_positions.items():
            current_price = position['current_price']
            entry_price = position['entry_price']
            shares = position['shares']
            
            # Calculate unrealized P&L
            unrealized_pnl = (current_price - entry_price) * shares
            
            # Check for trailing stop adjustment
            # If position is up significantly, adjust stop loss to lock in profit
            if unrealized_pnl > 0 and unrealized_pnl > position['dollar_risk'] * 1.5:
                # Move stop loss to break-even or better
                new_stop = max(position['stop_price'], entry_price)
                
                adjustments.append({
                    'symbol': symbol,
                    'action': 'adjust_stop',
                    'current_stop': position['stop_price'],
                    'new_stop': new_stop,
                    'reason': 'Trailing stop adjustment - lock in profit'
                })
            
            # Check for adding to winner (Ross Cameron's strategy)
            # If position is up and looking strong, consider adding more shares
            if (unrealized_pnl > 0 and 
                unrealized_pnl > position['dollar_risk'] and 
                self.cushion_achieved):
                
                # Calculate additional shares
                additional_shares = min(shares, self.calculate_position_size(
                    current_price, position['stop_price']))
                
                if additional_shares > 0:
                    adjustments.append({
                        'symbol': symbol,
                        'action': 'add_shares',
                        'current_shares': shares,
                        'additional_shares': additional_shares,
                        'reason': 'Add to winning position'
                    })
        
        return adjustments
    
    def should_continue_trading(self):
        """
        Determine if trading should continue for the day based on Ross Cameron's criteria.
        
        Returns:
        --------
        tuple: (should_continue, reason)
        """
        # Check for daily goal achievement
        daily_goal_amount = self.current_capital * self.daily_goal
        if self.daily_pnl >= daily_goal_amount:
            return True, "Daily goal achieved, trading can continue with full size"
        
        # Check for quarter goal achievement (profit cushion)
        quarter_goal_amount = self.current_capital * self.quarter_daily_goal
        if self.daily_pnl >= quarter_goal_amount:
            return True, "Profit cushion established, can trade with full size"
        
        # Check for daily loss limit
        daily_max_loss = self.current_capital * (self.daily_max_loss_pct / 100)
        if self.daily_pnl <= -daily_max_loss:
            return False, "Daily max loss reached, stop trading for the day"
        
        # Check for consecutive losses
        if self.consecutive_losses >= 3:
            return False, "Three consecutive losses, stop trading for the day"
        
        # Default: continue trading with reduced size
        return True, "Continue trading with reduced position size"
    
    def reset_daily_metrics(self):
        """
        Reset daily trading metrics at the start of a new trading day.
        """
        self.daily_pnl = 0.0
        self.cushion_achieved = False
        self.reduced_position_size = True
        self.consecutive_losses = 0
        
        logger.info("Daily metrics reset for new trading day")
    
    def get_performance_summary(self):
        """
        Get a summary of trading performance metrics.
        
        Returns:
        --------
        dict: Performance metrics
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_return_pct': (self.total_pnl / self.initial_capital) * 100,
            'win_rate': self.win_rate,
            'profit_loss_ratio': self.avg_win / abs(self.avg_loss) if self.avg_loss != 0 else 0,
            'max_drawdown': self.max_drawdown,
            'trade_count': len(self.trade_history),
            'consecutive_losses': self.consecutive_losses
        }
    
    def _update_performance_metrics(self, trade_pnl, is_win):
        """
        Update performance metrics after a trade.
        
        Parameters:
        -----------
        trade_pnl: float
            Profit/loss from the trade
        is_win: bool
            Whether the trade was a win
        """
        # Count total wins and losses
        wins = sum(1 for trade in self.trade_history if trade.get('realized_pnl', 0) > 0)
        losses = sum(1 for trade in self.trade_history if trade.get('realized_pnl', 0) <= 0)
        
        # Calculate win rate
        total_trades = wins + losses + 1  # Include current trade
        self.win_rate = wins / total_trades if is_win else (wins + 1) / total_trades
        
        # Calculate average win and loss
        win_amounts = [trade.get('realized_pnl', 0) for trade in self.trade_history 
                     if trade.get('realized_pnl', 0) > 0]
        loss_amounts = [trade.get('realized_pnl', 0) for trade in self.trade_history 
                      if trade.get('realized_pnl', 0) <= 0]
        
        if is_win:
            win_amounts.append(trade_pnl)
        else:
            loss_amounts.append(trade_pnl)
        
        self.avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0
        self.avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0