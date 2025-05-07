"""
Risk Manager Module

This module implements a risk management system for the trading strategy,
following Ross Cameron's approach of risking a small percentage of account
on each trade and maintaining strict risk control.
"""

import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RiskManager:
    """
    Manages risk for trading positions based on account size and risk parameters.
    
    This class implements position sizing, risk limits, and account management
    to ensure risk is controlled according to Ross Cameron's strategy.
    """
    
    def __init__(self, initial_capital=10000.0, max_risk_per_trade_pct=1.0,
               daily_max_loss_pct=3.0, profit_loss_ratio=2.0, max_open_positions=3):
        """
        Initialize the risk manager.
        
        Parameters:
        -----------
        initial_capital: float
            Starting capital amount
        max_risk_per_trade_pct: float
            Maximum percentage of account to risk on any trade
        daily_max_loss_pct: float
            Maximum percentage of account that can be lost in a day
        profit_loss_ratio: float
            Target profit-to-loss ratio for trades
        max_open_positions: int
            Maximum number of concurrent open positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.daily_max_loss_pct = daily_max_loss_pct
        self.profit_loss_ratio = profit_loss_ratio
        self.max_open_positions = max_open_positions
        
        # Track open positions
        self.open_positions = {}
        
        # Track daily performance
        self.daily_starting_capital = initial_capital
        self.daily_high_capital = initial_capital
        self.daily_low_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        
        # Track session dates
        self.session_date = datetime.now().strftime('%Y-%m-%d')
        
        # Additional flags for position sizing strategy
        self.reduced_position_size = True  # Start with reduced size
        self.cushion_achieved = False  # Profit cushion not achieved yet
        
        logger.info(f"Risk Manager initialized with capital: ${initial_capital:.2f}, " +
                   f"max risk per trade: {max_risk_per_trade_pct:.1f}%, " + 
                   f"daily max loss: {daily_max_loss_pct:.1f}%, " +
                   f"target P/L ratio: {profit_loss_ratio:.1f}")
    
    def reset_daily_metrics(self):
        """Reset daily performance metrics at the start of a new trading day."""
        self.daily_starting_capital = self.current_capital
        self.daily_high_capital = self.current_capital
        self.daily_low_capital = self.current_capital
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.session_date = datetime.now().strftime('%Y-%m-%d')
        
        # Reset position sizing flags
        self.reduced_position_size = True
        self.cushion_achieved = False
        
        logger.info(f"Daily metrics reset. Starting capital: ${self.daily_starting_capital:.2f}")
    
    def calculate_position_size(self, entry_price, stop_price, stock=None):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Parameters:
        -----------
        entry_price: float
            Planned entry price
        stop_price: float
            Planned stop loss price
        stock: Stock, optional
            Stock object for additional risk assessment
            
        Returns:
        --------
        int: Number of shares to trade
        """
        if entry_price <= 0 or stop_price <= 0 or entry_price <= stop_price:
            logger.warning("Invalid prices for position sizing")
            return 0
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_price
        
        # Calculate maximum dollar risk based on account size and risk percentage
        max_dollar_risk = self.current_capital * (self.max_risk_per_trade_pct / 100)
        
        # Reduce position size if using quarter size
        if self.reduced_position_size:
            max_dollar_risk *= 0.5  # Use half size until quarter daily goal is achieved
        
        # Calculate position size
        if risk_per_share > 0:
            shares = int(max_dollar_risk / risk_per_share)
        else:
            shares = 0
        
        # Adjust for stock-specific risk factors if stock object is provided
        if stock is not None:
            risk_level = stock.get_risk_level() if hasattr(stock, 'get_risk_level') else 5
            
            # Reduce position size for higher risk stocks
            if risk_level > 7:  # High risk
                shares = int(shares * 0.7)  # Reduce by 30%
            elif risk_level > 5:  # Moderate risk
                shares = int(shares * 0.85)  # Reduce by 15%
        
        # Ensure minimum shares
        shares = max(shares, 0)
        
        logger.info(f"Calculated position size: {shares} shares, " +
                   f"Risk per share: ${risk_per_share:.2f}, " +
                   f"Max dollar risk: ${max_dollar_risk:.2f}")
        
        return shares
    
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
            Number of shares to trade
            
        Returns:
        --------
        tuple: (bool, str) - Whether trade is valid and reason if not
        """
        # Check if prices are valid
        if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
            return False, "Invalid price levels"
        
        # Check if entry price is above stop price
        if entry_price <= stop_price:
            return False, "Entry price must be above stop price"
        
        # Check if target price is above entry price
        if target_price <= entry_price:
            return False, "Target price must be above entry price"
        
        # Check risk-reward ratio
        risk = entry_price - stop_price
        reward = target_price - entry_price
        
        if risk <= 0:
            return False, "Invalid risk calculation"
            
        actual_ratio = reward / risk
        
        if actual_ratio < self.profit_loss_ratio * 0.9:  # Allow 10% tolerance
            return False, f"Risk-reward ratio too low: {actual_ratio:.2f} (target: {self.profit_loss_ratio:.2f})"
        
        # Check if max number of open positions reached
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Maximum number of open positions reached ({self.max_open_positions})"
        
        # Check if symbol already in open positions
        if symbol in self.open_positions:
            return False, f"Position already open for {symbol}"
        
        # Check dollar risk
        dollar_risk = risk * shares
        max_dollar_risk = self.current_capital * (self.max_risk_per_trade_pct / 100)
        
        if dollar_risk > max_dollar_risk * 1.1:  # Allow 10% tolerance
            return False, f"Dollar risk too high: ${dollar_risk:.2f} (max: ${max_dollar_risk:.2f})"
        
        # Check daily max loss
        if self.daily_pnl < 0:
            daily_loss = abs(self.daily_pnl)
            max_daily_loss = self.daily_starting_capital * (self.daily_max_loss_pct / 100)
            
            # If we're approaching daily loss limit, reject new trades
            if daily_loss + dollar_risk > max_daily_loss:
                return False, f"Too close to daily loss limit: ${daily_loss:.2f} + ${dollar_risk:.2f} risk exceeds ${max_daily_loss:.2f}"
        
        # All checks passed
        return True, "Trade validated"
    
    def add_position(self, symbol, entry_price, stop_price, target_price, shares):
        """
        Add a new position to the risk manager.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        entry_price: float
            Entry price
        stop_price: float
            Stop loss price
        target_price: float
            Target price
        shares: int
            Number of shares
            
        Returns:
        --------
        bool: Success or failure
        """
        # Validate position parameters
        if entry_price <= 0 or stop_price <= 0 or target_price <= 0 or shares <= 0:
            logger.warning(f"Invalid position parameters for {symbol}")
            return False
        
        # Create position dictionary
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'entry_time': datetime.now(),
            'risk_per_share': entry_price - stop_price,
            'reward_per_share': target_price - entry_price,
            'dollar_risk': (entry_price - stop_price) * shares,
            'dollar_reward': (target_price - entry_price) * shares
        }
        
        # Add to open positions
        self.open_positions[symbol] = position
        
        logger.info(f"Added position for {symbol}: {shares} shares at ${entry_price:.2f}, " +
                   f"stop: ${stop_price:.2f}, target: ${target_price:.2f}")
        
        return True
    
    def close_position(self, symbol, exit_price, exit_reason='manual'):
        """
        Close an open position and update metrics.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        exit_price: float
            Exit price
        exit_reason: str
            Reason for closing position
            
        Returns:
        --------
        dict: Position details including P&L
        """
        if symbol not in self.open_positions:
            logger.warning(f"Cannot close position: {symbol} not found")
            return None
        
        # Get position
        position = self.open_positions[symbol]
        
        # Calculate P&L
        entry_price = position['entry_price']
        shares = position['shares']
        pnl = (exit_price - entry_price) * shares
        
        # Update position with exit details
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['pnl'] = pnl
        position['duration'] = (position['exit_time'] - position['entry_time']).total_seconds() / 60  # minutes
        
        # Update account metrics
        self.update_account_metrics(pnl, is_win=pnl > 0)
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        action = "Won" if pnl > 0 else "Lost"
        logger.info(f"{action} ${abs(pnl):.2f} on {symbol} ({exit_reason}). " +
                   f"Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, Shares: {shares}")
        
        return position
    
    def update_account_metrics(self, pnl, is_win=False):
        """
        Update account and performance metrics after a trade.
        
        Parameters:
        -----------
        pnl: float
            Profit/loss amount
        is_win: bool
            Whether the trade was a winner
            
        Returns:
        --------
        None
        """
        # Update capital
        self.current_capital += pnl
        
        # Update daily metrics
        self.daily_pnl += pnl
        
        if is_win:
            self.daily_win_count += 1
        else:
            self.daily_loss_count += 1
        
        # Update high/low water marks
        if self.current_capital > self.daily_high_capital:
            self.daily_high_capital = self.current_capital
        
        if self.current_capital < self.daily_low_capital:
            self.daily_low_capital = self.current_capital
        
        # Check if profit cushion achieved (at least 1% of account)
        cushion_threshold = self.daily_starting_capital * 0.01
        
        if self.daily_pnl >= cushion_threshold:
            self.cushion_achieved = True
            
            # Can use full position size once cushion achieved
            if self.reduced_position_size:
                self.reduced_position_size = False
                logger.info("Profit cushion achieved, switching to full position sizing")
        
        # Log account update
        logger.info(f"Account update: Capital: ${self.current_capital:.2f}, " +
                   f"Daily P&L: ${self.daily_pnl:.2f}, " +
                   f"Wins/Losses: {self.daily_win_count}/{self.daily_loss_count}")
    
    def get_open_positions(self):
        """
        Get all current open positions.
        
        Returns:
        --------
        dict: Dictionary of open positions
        """
        return self.open_positions
    
    def get_position_risk(self, symbol):
        """
        Get the current risk for a specific position.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        float: Dollar risk for the position, or 0 if position not found
        """
        if symbol in self.open_positions:
            return self.open_positions[symbol]['dollar_risk']
        return 0.0
    
    def get_total_risk(self):
        """
        Get the total current risk across all open positions.
        
        Returns:
        --------
        float: Total dollar risk
        """
        return sum(pos['dollar_risk'] for pos in self.open_positions.values())
    
    def check_max_daily_loss(self):
        """
        Check if the daily max loss limit has been reached.
        
        Returns:
        --------
        tuple: (bool, float) - Whether limit reached and remaining loss allowed
        """
        max_daily_loss = self.daily_starting_capital * (self.daily_max_loss_pct / 100)
        current_daily_loss = max(0, -self.daily_pnl)
        
        remaining_loss = max_daily_loss - current_daily_loss
        
        if remaining_loss <= 0:
            return True, 0.0
        
        return False, remaining_loss
    
    def get_performance_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
        --------
        dict: Performance metrics
        """
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'daily_starting_capital': self.daily_starting_capital,
            'daily_pnl': self.daily_pnl,
            'daily_return_pct': (self.current_capital / self.daily_starting_capital - 1) * 100,
            'daily_high_capital': self.daily_high_capital,
            'daily_low_capital': self.daily_low_capital,
            'daily_win_count': self.daily_win_count,
            'daily_loss_count': self.daily_loss_count,
            'win_rate': self.daily_win_count / (self.daily_win_count + self.daily_loss_count) 
                      if (self.daily_win_count + self.daily_loss_count) > 0 else 0,
            'open_positions_count': len(self.open_positions),
            'total_risk': self.get_total_risk(),
            'cushion_achieved': self.cushion_achieved,
            'reduced_position_size': self.reduced_position_size
        }