"""
Trade Manager

This module handles the execution and management of trades based on 
Ross Cameron's day trading strategy. It coordinates between scanning for opportunities,
risk management, and order execution.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, market_data_provider, scanner, risk_manager, condition_tracker, broker_api=None):
        """
        Initialize the trade manager.
        
        Parameters:
        -----------
        market_data_provider: MarketDataProvider
            Provider for market data
        scanner: StockScanner
            Scanner for finding trading opportunities
        risk_manager: RiskManager
            Manager for risk and position sizing
        condition_tracker: ConditionTracker
            Tracker for market conditions and patterns
        broker_api: object, optional
            API for trade execution with a broker
        """
        self.market_data = market_data_provider
        self.scanner = scanner
        self.risk_manager = risk_manager
        self.condition_tracker = condition_tracker
        self.broker = broker_api
        
        self.active_trades = {}
        self.pending_orders = {}
        self.completed_trades = []
        
        # Trading parameters
        self.min_price = 1.0
        self.max_price = 20.0
        self.min_gap_pct = 10.0
        self.min_rel_volume = 5.0
        self.max_float = 10_000_000
        
        # Trade execution flags
        self.is_trading_enabled = False
        self.is_simulated = broker_api is None
        
        # Performance tracking
        self.daily_stats = {
            'trades_taken': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'gross_pnl': 0.0
        }
    
    def start_trading_session(self):
        """
        Start a new trading session.
        """
        logger.info("Starting new trading session")
        
        # Reset daily metrics
        self.risk_manager.reset_daily_metrics()
        self.daily_stats = {
            'trades_taken': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'gross_pnl': 0.0
        }
        
        # Check market session
        self.condition_tracker.update_market_session()
        if not (self.condition_tracker.market_open or self.condition_tracker.pre_market):
            logger.warning("Market is not open, trading will be limited to simulation")
        
        # Enable trading
        self.is_trading_enabled = True
        
        logger.info("Trading session started")
    
    def stop_trading_session(self):
        """
        Stop the current trading session and close all open positions.
        """
        logger.info("Stopping trading session")
        
        # Disable trading
        self.is_trading_enabled = False
        
        # Close all open positions
        self._close_all_positions()
        
        # Log session summary
        self._log_session_summary()
        
        logger.info("Trading session stopped")
    
    def scan_for_opportunities(self):
        """
        Scan the market for trading opportunities based on Ross Cameron's strategy.
        
        Returns:
        --------
        list: Actionable trading opportunities
        """
        if not self.is_trading_enabled:
            logger.warning("Trading is disabled, scan aborted")
            return []
        
        logger.info("Scanning for trading opportunities")
        
        # Update market conditions
        self.condition_tracker.update_market_conditions()
        
        # Check if market is healthy for trading
        if not self.condition_tracker.is_market_healthy():
            logger.warning("Market conditions unfavorable, limiting trading")
            return []
        
        # First, scan for stocks meeting basic criteria
        momentum_stocks = self.scanner.scan_for_momentum_stocks(
            min_price=self.min_price,
            max_price=self.max_price,
            min_gap_pct=self.min_gap_pct,
            min_rel_volume=self.min_rel_volume,
            max_float=self.max_float
        )
        
        if not momentum_stocks:
            logger.info("No stocks meeting basic momentum criteria found")
            return []
        
        logger.info(f"Found {len(momentum_stocks)} stocks meeting basic momentum criteria")
        
        # Now, check for specific patterns
        self.condition_tracker.scan_for_trading_conditions(
            momentum_stocks,
            min_gap_pct=self.min_gap_pct,
            min_rel_volume=self.min_rel_volume,
            max_float=self.max_float
        )
        
        # Get actionable opportunities
        opportunities = self.condition_tracker.get_actionable_stocks(max_stocks=5)
        
        if not opportunities:
            logger.info("No actionable trading opportunities found")
            return []
        
        logger.info(f"Found {len(opportunities)} actionable trading opportunities")
        return opportunities
    
    def evaluate_opportunity(self, stock):
        """
        Evaluate a trading opportunity and determine entry parameters.
        
        Parameters:
        -----------
        stock: Stock
            Stock object with pattern detection
            
        Returns:
        --------
        dict: Trade parameters if opportunity is valid, None otherwise
        """
        if not self.is_trading_enabled:
            logger.warning("Trading is disabled, evaluation aborted")
            return None
        
        # Determine pattern type
        if stock.has_bull_flag:
            pattern_type = 'bull_flag'
        elif stock.has_new_high_breakout:
            pattern_type = 'new_high_breakout'
        elif stock.has_micro_pullback:
            pattern_type = 'micro_pullback'
        else:
            logger.warning(f"No actionable pattern detected for {stock.symbol}")
            return None
        
        logger.info(f"Evaluating {pattern_type} opportunity for {stock.symbol}")
        
        # Get entry, stop, and target prices
        entry_price = stock.get_optimal_entry()
        stop_price = stock.get_optimal_stop_loss()
        target_price = stock.get_optimal_target()
        
        # Check if prices are valid
        if entry_price is None or stop_price is None or target_price is None:
            logger.warning(f"Invalid price levels for {stock.symbol}")
            return None
        
        # Calculate position size
        shares = self.risk_manager.calculate_position_size(entry_price, stop_price, stock)
        
        # Check if position size is valid
        if shares <= 0:
            logger.warning(f"Invalid position size for {stock.symbol}")
            return None
        
        # Validate the trade
        is_valid, reason = self.risk_manager.validate_trade(
            stock.symbol, entry_price, stop_price, target_price, shares
        )
        
        if not is_valid:
            logger.warning(f"Trade validation failed for {stock.symbol}: {reason}")
            return None
        
        # Create trade parameters
        trade_params = {
            'symbol': stock.symbol,
            'pattern': pattern_type,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'risk_per_share': entry_price - stop_price,
            'reward_per_share': target_price - entry_price,
            'dollar_risk': (entry_price - stop_price) * shares,
            'dollar_reward': (target_price - entry_price) * shares,
            'profit_loss_ratio': (target_price - entry_price) / (entry_price - stop_price),
            'timestamp': datetime.now()
        }
        
        logger.info(f"Trade opportunity validated for {stock.symbol}: {shares} shares, " +
                   f"Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
        
        return trade_params
    
    def execute_trade(self, trade_params):
        """
        Execute a trade based on the given parameters.
        
        Parameters:
        -----------
        trade_params: dict
            Trade parameters from evaluate_opportunity
            
        Returns:
        --------
        dict: Executed trade information
        """
        if not self.is_trading_enabled:
            logger.warning("Trading is disabled, execution aborted")
            return None
        
        symbol = trade_params['symbol']
        entry_price = trade_params['entry_price']
        shares = trade_params['shares']
        
        logger.info(f"Executing trade for {symbol}: {shares} shares at ${entry_price:.2f}")
        
        # Check if we're in simulation mode
        if self.is_simulated:
            # Simulate trade execution
            executed_price = entry_price
            executed_shares = shares
            commission = 0.0
            order_id = f"SIM_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            logger.info(f"Simulated trade executed for {symbol}: {executed_shares} shares at ${executed_price:.2f}")
        else:
            # Execute trade with broker API
            try:
                # Submit order
                order_result = self.broker.submit_order(
                    symbol=symbol,
                    quantity=shares,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=entry_price
                )
                
                # Get execution details
                order_id = order_result.get('id', '')
                executed_price = order_result.get('average_price', entry_price)
                executed_shares = order_result.get('filled_quantity', 0)
                commission = order_result.get('commission', 0.0)
                
                if executed_shares <= 0:
                    logger.warning(f"Order not filled for {symbol}")
                    return None
                
                logger.info(f"Trade executed for {symbol}: {executed_shares} shares at ${executed_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                return None
        
        # Update trade parameters with execution details
        trade_params['executed_price'] = executed_price
        trade_params['executed_shares'] = executed_shares
        trade_params['commission'] = commission
        trade_params['order_id'] = order_id
        trade_params['execution_time'] = datetime.now()
        trade_params['status'] = 'open'
        
        # Add to active trades
        self.active_trades[symbol] = trade_params
        
        # Add position to risk manager
        self.risk_manager.add_position(
            symbol=symbol,
            entry_price=executed_price,
            stop_price=trade_params['stop_price'],
            target_price=trade_params['target_price'],
            shares=executed_shares
        )
        
        # Update daily stats
        self.daily_stats['trades_taken'] += 1
        
        # Return executed trade information
        return trade_params
    
    def manage_active_trades(self):
        """
        Manage active trades based on current market conditions and risk management rules.
        
        Returns:
        --------
        list: Actions taken on active trades
        """
        if not self.active_trades:
            return []
        
        actions_taken = []
        symbols_to_remove = []
        
        for symbol, trade in self.active_trades.items():
            # Get current price
            current_price = self.market_data.get_current_price(symbol)
            
            if current_price is None:
                logger.warning(f"Could not get current price for {symbol}")
                continue
            
            # Update trade with current price
            trade['current_price'] = current_price
            
            # Calculate unrealized P&L
            entry_price = trade['executed_price']
            shares = trade['executed_shares']
            unrealized_pnl = (current_price - entry_price) * shares - trade.get('commission', 0.0)
            trade['unrealized_pnl'] = unrealized_pnl
            
            # Check stop loss
            if current_price <= trade['stop_price']:
                action = {
                    'symbol': symbol,
                    'action': 'exit',
                    'reason': 'Stop loss hit',
                    'price': current_price,
                    'pnl': unrealized_pnl
                }
                
                # Exit the trade
                self._exit_trade(symbol, current_price, 'stop_loss')
                symbols_to_remove.append(symbol)
                actions_taken.append(action)
                
                logger.info(f"Stop loss hit for {symbol} at ${current_price:.2f}, P&L: ${unrealized_pnl:.2f}")
                continue
            
            # Check target price
            if current_price >= trade['target_price']:
                action = {
                    'symbol': symbol,
                    'action': 'exit',
                    'reason': 'Target reached',
                    'price': current_price,
                    'pnl': unrealized_pnl
                }
                
                # Exit the trade
                self._exit_trade(symbol, current_price, 'target_reached')
                symbols_to_remove.append(symbol)
                actions_taken.append(action)
                
                logger.info(f"Target reached for {symbol} at ${current_price:.2f}, P&L: ${unrealized_pnl:.2f}")
                continue
            
            # Check for trailing stop adjustments
            if unrealized_pnl > 0 and unrealized_pnl > trade['dollar_risk']:
                # Calculate new stop loss (break-even or better)
                new_stop = max(trade['stop_price'], entry_price)
                
                # Only update if it's higher than current stop
                if new_stop > trade['stop_price']:
                    action = {
                        'symbol': symbol,
                        'action': 'adjust_stop',
                        'old_stop': trade['stop_price'],
                        'new_stop': new_stop,
                        'reason': 'Trailing stop'
                    }
                    
                    # Update stop loss
                    trade['stop_price'] = new_stop
                    actions_taken.append(action)
                    
                    logger.info(f"Trailing stop adjusted for {symbol}: ${trade['stop_price']:.2f} -> ${new_stop:.2f}")
            
            # Check for adding to winner (Ross Cameron's strategy)
            if (unrealized_pnl > 0 and 
                unrealized_pnl > trade['dollar_risk'] and 
                self.risk_manager.cushion_achieved and
                'add_shares_processed' not in trade):
                
                # Calculate additional shares
                additional_shares = min(shares, self.risk_manager.calculate_position_size(
                    current_price, trade['stop_price']))
                
                if additional_shares > 0:
                    action = {
                        'symbol': symbol,
                        'action': 'add_shares',
                        'current_shares': shares,
                        'additional_shares': additional_shares,
                        'price': current_price,
                        'reason': 'Add to winner'
                    }
                    
                    # Execute additional shares
                    if not self.is_simulated and self.broker:
                        try:
                            order_result = self.broker.submit_order(
                                symbol=symbol,
                                quantity=additional_shares,
                                side='buy',
                                type='limit',
                                time_in_force='day',
                                limit_price=current_price
                            )
                            
                            # Update trade with additional shares
                            executed_additional = order_result.get('filled_quantity', 0)
                            additional_commission = order_result.get('commission', 0.0)
                            
                            if executed_additional > 0:
                                # Update trade details
                                new_total_shares = shares + executed_additional
                                new_avg_price = ((entry_price * shares) + 
                                               (current_price * executed_additional)) / new_total_shares
                                
                                trade['executed_shares'] = new_total_shares
                                trade['executed_price'] = new_avg_price
                                trade['commission'] += additional_commission
                                trade['add_shares_processed'] = True
                                
                                logger.info(f"Added {executed_additional} shares to {symbol} at ${current_price:.2f}")
                        except Exception as e:
                            logger.error(f"Error adding shares to {symbol}: {e}")
                    else:
                        # Simulate adding shares
                        new_total_shares = shares + additional_shares
                        new_avg_price = ((entry_price * shares) + 
                                       (current_price * additional_shares)) / new_total_shares
                        
                        trade['executed_shares'] = new_total_shares
                        trade['executed_price'] = new_avg_price
                        trade['add_shares_processed'] = True
                        
                        logger.info(f"Simulated adding {additional_shares} shares to {symbol} at ${current_price:.2f}")
                    
                    actions_taken.append(action)
        
        # Remove closed positions
        for symbol in symbols_to_remove:
            if symbol in self.active_trades:
                del self.active_trades[symbol]
        
        return actions_taken
    
    def _exit_trade(self, symbol, exit_price, exit_reason):
        """
        Exit a trade and update tracking information.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        exit_price: float
            Exit price
        exit_reason: str
            Reason for exiting the trade
            
        Returns:
        --------
        dict: Closed trade information
        """
        if symbol not in self.active_trades:
            logger.warning(f"Trade not found for {symbol}")
            return None
        
        trade = self.active_trades[symbol]
        
        # Execute sell order
        if not self.is_simulated and self.broker:
            try:
                order_result = self.broker.submit_order(
                    symbol=symbol,
                    quantity=trade['executed_shares'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Get execution details
                executed_exit_price = order_result.get('average_price', exit_price)
                exit_commission = order_result.get('commission', 0.0)
                
                logger.info(f"Sold {trade['executed_shares']} shares of {symbol} at ${executed_exit_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error selling {symbol}: {e}")
                executed_exit_price = exit_price
                exit_commission = 0.0
        else:
            # Simulate sell execution
            executed_exit_price = exit_price
            exit_commission = 0.0
            
            logger.info(f"Simulated selling {trade['executed_shares']} shares of {symbol} at ${executed_exit_price:.2f}")
        
        # Calculate realized P&L
        entry_price = trade['executed_price']
        shares = trade['executed_shares']
        commission = trade.get('commission', 0.0) + exit_commission
        realized_pnl = (executed_exit_price - entry_price) * shares - commission
        
        # Update trade details
        trade['exit_price'] = executed_exit_price
        trade['exit_time'] = datetime.now()
        trade['exit_reason'] = exit_reason
        trade['realized_pnl'] = realized_pnl
        trade['commission'] = commission
        trade['status'] = 'closed'
        
        # Calculate duration
        duration = trade['exit_time'] - trade['execution_time']
        trade['duration_seconds'] = duration.total_seconds()
        
        # Update risk manager
        self.risk_manager.close_position(symbol, executed_exit_price, exit_reason)
        
        # Update daily stats
        if realized_pnl > 0:
            self.daily_stats['winning_trades'] += 1
            self.daily_stats['total_profit'] += realized_pnl
        else:
            self.daily_stats['losing_trades'] += 1
            self.daily_stats['total_loss'] += realized_pnl
            
        self.daily_stats['gross_pnl'] += realized_pnl
        
        # Add to completed trades
        self.completed_trades.append(trade)
        
        return trade
    
    def _close_all_positions(self):
        """
        Close all open positions.
        """
        logger.info("Closing all open positions")
        
        symbols = list(self.active_trades.keys())
        
        for symbol in symbols:
            # Get current price
            current_price = self.market_data.get_current_price(symbol)
            
            if current_price is None:
                logger.warning(f"Could not get current price for {symbol}, using last known price")
                current_price = self.active_trades[symbol].get('current_price', 
                                                             self.active_trades[symbol]['executed_price'])
            
            # Exit the trade
            self._exit_trade(symbol, current_price, 'session_end')
            
            logger.info(f"Closed position for {symbol} at ${current_price:.2f}")
    
    def _log_session_summary(self):
        """
        Log a summary of the trading session.
        """
        total_trades = self.daily_stats['trades_taken']
        winning_trades = self.daily_stats['winning_trades']
        losing_trades = self.daily_stats['losing_trades']
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_pnl = self.daily_stats['gross_pnl']
        
        logger.info("=== Trading Session Summary ===")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades} ({win_rate:.2%})")
        logger.info(f"Losing Trades: {losing_trades} ({1-win_rate:.2%})")
        logger.info(f"Gross P&L: ${gross_pnl:.2f}")
        logger.info("==============================")
    
    def get_trading_status(self):
        """
        Get the current trading status.
        
        Returns:
        --------
        dict: Trading status information
        """
        return {
            'is_trading_enabled': self.is_trading_enabled,
            'is_simulated': self.is_simulated,
            'market_open': self.condition_tracker.market_open if self.condition_tracker else False,
            'pre_market': self.condition_tracker.pre_market if self.condition_tracker else False,
            'strong_market': self.condition_tracker.strong_market if self.condition_tracker else False,
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'daily_pnl': self.daily_stats['gross_pnl'],
            'cushion_achieved': self.risk_manager.cushion_achieved if self.risk_manager else False,
            'reduced_position_size': self.risk_manager.reduced_position_size if self.risk_manager else True
        }