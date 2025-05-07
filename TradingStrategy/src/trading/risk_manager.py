"""
Trade Manager

This module handles the execution and management of trades based on the 
trading strategy. It coordinates between scanning for opportunities,
risk management, and order execution.
"""

import os
import logging
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

from src.entry.candlestick import CandlestickPatterns
from src.indicators.moving_averages import MovingAverages
from src.indicators.macd import MACD

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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
        
        # Initialize pattern detection and indicators
        self.candlestick_patterns = CandlestickPatterns()
        self.moving_averages = MovingAverages()
        self.macd = MACD()
        
        # Trading parameters
        self.min_price = 2.0
        self.max_price = 20.0
        self.min_gap_pct = 10.0
        self.min_rel_volume = 5.0
        self.max_float = 20_000_000
        
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
        
        logger.info("Trade Manager initialized")
    
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
        
        # Clear active trades at the start of a new session
        self.active_trades = {}
        self.pending_orders = {}
        
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
        Scan the market for trading opportunities based on the strategy conditions.
        
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
        
        # Scan for stocks meeting all 5 conditions
        momentum_stocks = self.scanner.scan_for_momentum_stocks(
            min_price=self.min_price,
            max_price=self.max_price,
            min_gap_pct=self.min_gap_pct,
            min_rel_volume=self.min_rel_volume,
            max_float=self.max_float
        )
        
        if not momentum_stocks:
            logger.info("No stocks meeting all 5 criteria found")
            return []
        
        logger.info(f"Found {len(momentum_stocks)} stocks meeting all 5 criteria")
        
        # Track these stocks for entry patterns
        tracked_stocks = self.condition_tracker.scan_for_trading_conditions(
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
        
        # Enhance opportunities with technical indicators
        enhanced_opportunities = []
        
        for stock in opportunities:
            # Get intraday data
            intraday_data = self.market_data.get_intraday_data(
                stock.symbol, interval='1m', lookback_days=1
            )
            
            if intraday_data.empty:
                continue
            
            # Add EMAs
            intraday_data = self.moving_averages.add_moving_averages(intraday_data)
            
            # Add MACD
            intraday_data = self.macd.add_macd(intraday_data)
            
            # Check if price is above key EMAs
            price_above_emas = self.moving_averages.is_price_above_key_emas(intraday_data)
            
            # Check for bullish MACD
            bullish_macd = self.macd.is_bullish_momentum(intraday_data)
            
            # Only include stocks with favorable indicators
            if price_above_emas and bullish_macd:
                stock.set_price_history(intraday_data)
                enhanced_opportunities.append(stock)
                logger.info(f"Enhanced opportunity: {stock.symbol} - Above key EMAs: {price_above_emas}, Bullish MACD: {bullish_macd}")
            else:
                logger.info(f"Filtered out {stock.symbol} - Above key EMAs: {price_above_emas}, Bullish MACD: {bullish_macd}")
        
        return enhanced_opportunities
    
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
        
        # Get pattern information
        pattern_signals = self.candlestick_patterns.detect_entry_signal(stock.price_history)
        
        if not pattern_signals:
            logger.warning(f"No actionable pattern detected for {stock.symbol}")
            return None
        
        # Determine primary pattern type (prioritize based on strategy)
        if 'bull_flag' in pattern_signals:
            pattern_type = 'bull_flag'
        elif 'new_high_breakout' in pattern_signals:
            pattern_type = 'new_high_breakout'
        elif 'micro_pullback' in pattern_signals:
            pattern_type = 'micro_pullback'
        else:
            logger.warning(f"No actionable pattern detected for {stock.symbol}")
            return None
        
        logger.info(f"Evaluating {pattern_type} opportunity for {stock.symbol}")
        
        # Get entry, stop, and target prices
        entry_price = self.candlestick_patterns.get_optimal_entry_price(stock.price_history, pattern_type)
        stop_price = self.candlestick_patterns.get_optimal_stop_price(stock.price_history, pattern_type)
        
        # Ensure stop is not too tight
        stop_pct = (entry_price - stop_price) / entry_price * 100
        if stop_pct < 2.0:
            stop_price = entry_price * 0.98  # At least 2% below entry
            logger.info(f"Adjusted stop-loss to 2% below entry: ${stop_price:.2f}")
        
        # Calculate target price based on profit-loss ratio
        target_price = self.candlestick_patterns.get_optimal_target_price(
            entry_price, stop_price, self.risk_manager.profit_loss_ratio
        )
        
        # Check if prices are valid
        if entry_price is None or stop_price is None or target_price is None:
            logger.warning(f"Invalid price levels for {stock.symbol}")
            return None
        
        # Recalculate to ensure correct profit-to-loss ratio
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        profit_loss_ratio = reward_per_share / risk_per_share
        
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
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'dollar_risk': risk_per_share * shares,
            'dollar_reward': reward_per_share * shares,
            'profit_loss_ratio': profit_loss_ratio,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Trade opportunity validated for {stock.symbol}: {shares} shares, " +
                   f"Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}, " +
                   f"Risk: ${risk_per_share:.2f}/share, Reward: ${reward_per_share:.2f}/share, " +
                   f"P/L Ratio: {profit_loss_ratio:.2f}")
        
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
                    side='BUY',
                    order_type='LIMIT',
                    time_in_force='DAY',
                    limit_price=entry_price
                )
                
                if not order_result:
                    logger.error(f"Failed to submit order for {symbol}")
                    return None
                
                # Get execution details
                order_id = order_result['id']
                executed_price = entry_price  # Assume limit price execution
                executed_shares = shares
                commission = 0.0  # Will be updated later with actual commission
                
                logger.info(f"Trade executed for {symbol}: {executed_shares} shares at ${executed_price:.2f}")
                
                # Set stop loss and take profit orders
                self._set_exit_orders(symbol, trade_params)
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {str(e)}")
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
    
    def _set_exit_orders(self, symbol, trade_params):
        """
        Set stop loss and take profit orders.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        trade_params: dict
            Trade parameters
            
        Returns:
        --------
        tuple: (stop_order_id, target_order_id)
        """
        if self.is_simulated or not self.broker:
            return None, None
        
        try:
            shares = trade_params['executed_shares']
            stop_price = trade_params['stop_price']
            target_price = trade_params['target_price']
            
            # Set stop loss order
            stop_order = self.broker.submit_order(
                symbol=symbol,
                quantity=shares,
                side='SELL',
                order_type='STOP',
                time_in_force='GTC',  # Good Till Canceled
                stop_price=stop_price
            )
            
            if stop_order:
                stop_order_id = stop_order['id']
                logger.info(f"Stop loss order placed at ${stop_price:.2f}")
                trade_params['stop_order_id'] = stop_order_id
            else:
                logger.warning(f"Failed to place stop loss order for {symbol}")
                stop_order_id = None
            
            # Set take profit order
            profit_order = self.broker.submit_order(
                symbol=symbol,
                quantity=shares,
                side='SELL',
                order_type='LIMIT',
                time_in_force='GTC',  # Good Till Canceled
                limit_price=target_price
            )
            
            if profit_order:
                profit_order_id = profit_order['id']
                logger.info(f"Take profit order placed at ${target_price:.2f}")
                trade_params['profit_order_id'] = profit_order_id
            else:
                logger.warning(f"Failed to place take profit order for {symbol}")
                profit_order_id = None
            
            return stop_order_id, profit_order_id
            
        except Exception as e:
            logger.error(f"Error setting exit orders for {symbol}: {str(e)}")
            return None, None
    
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
            # Only trail once we've moved at least halfway to target
            halfway_to_target = entry_price + ((trade['target_price'] - entry_price) / 2)
            
            if current_price >= halfway_to_target:
                # Calculate new stop loss based on how close we are to target
                progress_to_target = (current_price - entry_price) / (trade['target_price'] - entry_price)
                
                # As we get closer to target, trail tighter
                if progress_to_target >= 0.8:  # 80% of the way to target
                    # Trail to 75% of gains (aggressive trailing)
                    new_stop = entry_price + ((current_price - entry_price) * 0.75)
                elif progress_to_target >= 0.5:  # 50-80% of the way
                    # Trail to breakeven plus 50% of gains
                    new_stop = entry_price + ((current_price - entry_price) * 0.5)
                else:  # Just reached halfway
                    # Trail to breakeven
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
                    
                    # If using a broker, update the stop loss order
                    if not self.is_simulated and self.broker and 'stop_order_id' in trade:
                        try:
                            # Cancel the old stop order
                            self.broker.cancel_order(trade['stop_order_id'])
                            
                            # Place a new stop order
                            new_stop_order = self.broker.submit_order(
                                symbol=symbol,
                                quantity=shares,
                                side='SELL',
                                order_type='STOP',
                                time_in_force='GTC',
                                stop_price=new_stop
                            )
                            
                            if new_stop_order:
                                trade['stop_order_id'] = new_stop_order['id']
                        except Exception as e:
                            logger.error(f"Error updating stop loss for {symbol}: {str(e)}")
                    
                    actions_taken.append(action)
                    logger.info(f"Trailing stop adjusted for {symbol}: ${trade['stop_price']:.2f} -> ${new_stop:.2f}")
            
            # Check for adding to winner
            # Only if we've reached halfway to target and have a profit cushion
            if (current_price >= halfway_to_target and 
                self.risk_manager.cushion_achieved and
                'add_shares_processed' not in trade):
                
                # Calculate additional shares - limit to 50% of original position
                additional_shares = min(shares // 2, self.risk_manager.calculate_position_size(
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
                            # Submit order for additional shares
                            add_order = self.broker.submit_order(
                                symbol=symbol,
                                quantity=additional_shares,
                                side='BUY',
                                order_type='LIMIT',
                                time_in_force='DAY',
                                limit_price=current_price
                            )
                            
                            if add_order:
                                # Update trade with additional shares
                                new_total_shares = shares + additional_shares
                                new_avg_price = ((entry_price * shares) + 
                                               (current_price * additional_shares)) / new_total_shares
                                
                                trade['executed_shares'] = new_total_shares
                                trade['executed_price'] = new_avg_price
                                trade['add_shares_processed'] = True
                                
                                # Update exit orders
                                if 'stop_order_id' in trade:
                                    self.broker.cancel_order(trade['stop_order_id'])
                                if 'profit_order_id' in trade:
                                    self.broker.cancel_order(trade['profit_order_id'])
                                
                                # Set new exit orders
                                self._set_exit_orders(symbol, trade)
                                
                                logger.info(f"Added {additional_shares} shares to {symbol} at ${current_price:.2f}")
                            else:
                                logger.warning(f"Failed to add shares to {symbol}")
                        except Exception as e:
                            logger.error(f"Error adding shares to {symbol}: {str(e)}")
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
            logger.warning(f"Position not found: {symbol}")
            return None
        
        trade = self.active_trades[symbol]
        
        # If using a broker, cancel existing exit orders
        if not self.is_simulated and self.broker:
            try:
                # Cancel stop loss and take profit orders
                if 'stop_order_id' in trade:
                    self.broker.cancel_order(trade['stop_order_id'])
                
                if 'profit_order_id' in trade:
                    self.broker.cancel_order(trade['profit_order_id'])
                
                # Execute sell order
                sell_order = self.broker.submit_order(
                    symbol=symbol,
                    quantity=trade['executed_shares'],
                    side='SELL',
                    order_type='MARKET',  # Use market order to ensure execution
                    time_in_force='DAY'
                )
                
                if sell_order:
                    executed_exit_price = sell_order.get('avg_fill_price', exit_price)
                    commission = sell_order.get('commission', 0.0)
                    
                    logger.info(f"Sold {trade['executed_shares']} shares of {symbol} at ${executed_exit_price:.2f}")
                else:
                    executed_exit_price = exit_price
                    commission = 0.0
                    
                    logger.warning(f"Failed to execute sell order for {symbol}, using estimated exit price")
                
            except Exception as e:
                logger.error(f"Error selling {symbol}: {str(e)}")
                executed_exit_price = exit_price
                commission = 0.0
        else:
            # Simulate sell execution
            executed_exit_price = exit_price
            commission = 0.0
            
            logger.info(f"Simulated selling {trade['executed_shares']} shares of {symbol} at ${executed_exit_price:.2f}")
        
        # Calculate realized P&L
        entry_price = trade['executed_price']
        shares = trade['executed_shares']
        total_commission = trade.get('commission', 0.0) + commission
        realized_pnl = (executed_exit_price - entry_price) * shares - total_commission
        
        # Update trade details
        trade['exit_price'] = executed_exit_price
        trade['exit_time'] = datetime.now()
        trade['exit_reason'] = exit_reason
        trade['realized_pnl'] = realized_pnl
        trade['commission'] = total_commission
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