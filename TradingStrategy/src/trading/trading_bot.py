"""
Trading Bot - Ross Cameron Strategy Implementation

This module implements the complete trading flow:
1. Check 5 conditions
2. Detect entry patterns (Bull Flag, First Pullback, etc.)
3. Manage trades with proper exits
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

from src.conditions.condition_checker import ConditionChecker
from src.patterns.pattern_detector import PatternDetector
from src.trading.risk_manager import RiskManager
from src.utils.logger import setup_logger
from src.data.ib_connector import IBConnector
from src.indicators.macd import MACD
from src.indicators.moving_averages import MovingAverages


class TradingBot:
    def __init__(self, initial_capital=10000.0):
        """Initialize the trading bot with Ross Cameron's strategy."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades = {}
        self.completed_trades = []
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.trading_started_today = False
        
        # Initialize components
        self.condition_checker = ConditionChecker()
        self.pattern_detector = PatternDetector(patterns_to_load=[
            'bull_flag',
            'first_pullback',
            'micro_pullback',
            'flat_top_breakout',
            'candle_over_candle'  # Part of micro pullback
        ])
        
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Initialize indicators
        self.macd = MACD()
        self.moving_averages = MovingAverages()
        
        # IB Connection
        self.ib_connector = None
        
        # Set up logging
        self.logger = setup_logger('trading_bot')
        self.logger.info(f"Trading Bot initialized with ${initial_capital:.2f}")
    
    def connect_to_ib(self):
        """Connect to Interactive Brokers."""
        self.ib_connector = IBConnector()
        if not self.ib_connector.connect():
            self.logger.error("Failed to connect to Interactive Brokers")
            return False
        return True
    
    def start_trading_session(self):
        """Start a new trading session."""
        self.logger.info("Starting new trading session")
        
        # Reset daily metrics
        self.risk_manager.reset_daily_metrics()
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.trading_started_today = False
        
        # Connect to IB if not connected
        if not self.ib_connector or not self.ib_connector.connected:
            if not self.connect_to_ib():
                return False
        
        return True
    
    def scan_stocks(self, watch_list):
        """
        Scan stocks for those meeting all 5 conditions.
        
        Parameters:
        -----------
        watch_list: list
            List of stock symbols to scan
            
        Returns:
        --------
        list: Stocks meeting all 5 conditions
        """
        qualified_stocks = []
        
        for symbol in watch_list:
            try:
                # Get stock data
                stock_data = self._get_stock_data(symbol)
                if not stock_data:
                    continue
                
                # Check all 5 conditions
                all_conditions_met, condition_results = self.condition_checker.check_all_conditions(stock_data)
                
                if all_conditions_met:
                    qualified_stocks.append({
                        'symbol': symbol,
                        'data': stock_data,
                        'conditions': condition_results
                    })
                    self.logger.info(f"{symbol} meets all 5 conditions")
                else:
                    failed_conditions = [k for k, v in condition_results.items() if not v]
                    self.logger.debug(f"{symbol} failed conditions: {failed_conditions}")
                    
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                
        return qualified_stocks
    
    def find_entry_signals(self, qualified_stocks):
        """
        Look for entry patterns in stocks that meet all 5 conditions.
        
        Parameters:
        -----------
        qualified_stocks: list
            Stocks that passed the 5 conditions check
            
        Returns:
        --------
        list: Entry opportunities with pattern details
        """
        entry_opportunities = []
        
        for stock_info in qualified_stocks:
            symbol = stock_info['symbol']
            
            try:
                # Get intraday data for pattern detection
                intraday_data = self._get_intraday_data(symbol)
                if intraday_data.empty:
                    continue
                
                # Add technical indicators
                intraday_data = self._add_indicators(intraday_data)
                
                # Detect patterns
                patterns = self.pattern_detector.get_bullish_patterns(intraday_data)
                
                # Filter for high confidence patterns
                high_confidence_patterns = [p for p in patterns if p.get('confidence', 0) >= 70]
                
                if high_confidence_patterns:
                    # Sort by confidence
                    high_confidence_patterns.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Take the best pattern
                    best_pattern = high_confidence_patterns[0]
                    
                    entry_opportunities.append({
                        'symbol': symbol,
                        'pattern': best_pattern,
                        'stock_data': stock_info['data'],
                        'intraday_data': intraday_data
                    })
                    
                    self.logger.info(f"Entry signal for {symbol}: {best_pattern['pattern']} "
                                   f"({best_pattern['confidence']:.1f}% confidence)")
                    
            except Exception as e:
                self.logger.error(f"Error finding entry signals for {symbol}: {e}")
                
        return entry_opportunities
    
    def execute_trades(self, entry_opportunities):
        """
        Execute trades based on entry signals.
        
        Parameters:
        -----------
        entry_opportunities: list
            Valid entry opportunities
        """
        for opportunity in entry_opportunities:
            try:
                symbol = opportunity['symbol']
                pattern = opportunity['pattern']
                
                # Skip if we already have a position in this stock
                if symbol in self.active_trades:
                    self.logger.info(f"Already have position in {symbol}, skipping")
                    continue
                
                # Skip if we've reached max positions
                if len(self.active_trades) >= self.risk_manager.max_open_positions:
                    self.logger.info("Maximum positions reached, skipping new trades")
                    break
                
                # Get entry parameters from pattern
                entry_price = pattern['entry_price']
                stop_price = pattern['stop_price']
                
                # Calculate position size
                shares = self.risk_manager.calculate_position_size(entry_price, stop_price)
                
                if shares <= 0:
                    self.logger.warning(f"Invalid position size for {symbol}")
                    continue
                
                # Validate the trade
                is_valid, reason = self.risk_manager.validate_trade(
                    symbol, entry_price, stop_price, 
                    pattern.get('target_price', entry_price * 1.02), shares
                )
                
                if not is_valid:
                    self.logger.warning(f"Trade validation failed for {symbol}: {reason}")
                    continue
                
                # Execute the trade
                trade = self._execute_trade(symbol, pattern, shares)
                
                if trade:
                    self.active_trades[symbol] = trade
                    self.logger.info(f"Trade executed: {symbol} - {shares} shares at ${entry_price:.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error executing trade for {opportunity['symbol']}: {e}")
    
    def manage_active_trades(self):
        """
        Manage active trades - check for exits based on Ross Cameron's strategy.
        
        Returns:
        --------
        list: Actions taken on trades
        """
        if not self.active_trades:
            return []
        
        actions_taken = []
        trades_to_close = []
        
        for symbol, trade in self.active_trades.items():
            try:
                # Get current data
                current_data = self._get_current_data(symbol)
                if not current_data:
                    continue
                
                current_candle = current_data.iloc[-1]
                
                # Check exit conditions based on Ross Cameron's strategy
                exit_signal = self._check_exit_conditions(trade, current_data)
                
                if exit_signal:
                    # Exit the trade
                    exit_result = self._exit_trade(trade, current_candle, exit_signal)
                    trades_to_close.append(symbol)
                    actions_taken.append(exit_result)
                else:
                    # Check if we should adjust the stop (trailing)
                    adjustment = self._check_stop_adjustment(trade, current_data)
                    if adjustment:
                        actions_taken.append(adjustment)
                        
            except Exception as e:
                self.logger.error(f"Error managing trade for {symbol}: {e}")
        
        # Remove closed trades
        for symbol in trades_to_close:
            del self.active_trades[symbol]
            
        return actions_taken
    
    def _check_exit_conditions(self, trade, current_data):
        """
        Check exit conditions based on Ross Cameron's strategy.
        
        Parameters:
        -----------
        trade: dict
            Active trade details
        current_data: DataFrame
            Current price and indicator data
            
        Returns:
        --------
        dict or None: Exit signal if conditions are met
        """
        current_candle = current_data.iloc[-1]
        
        # 1. Stop Loss Check
        if current_candle['low'] <= trade['stop_price']:
            return {
                'reason': 'stop_loss',
                'price': trade['stop_price'],
                'candle': current_candle
            }
        
        # 2. Target Check (if set)
        if 'target_price' in trade and current_candle['high'] >= trade['target_price']:
            return {
                'reason': 'target_reached',
                'price': trade['target_price'],
                'candle': current_candle
            }
        
        # 3. Reversal Pattern Check (Ross Cameron's exit signals)
        exit_patterns = self._check_exit_patterns(current_data)
        
        if exit_patterns:
            # Confirmed exit signal
            pattern = exit_patterns[0]  # Take the first/strongest signal
            
            return {
                'reason': 'exit_pattern',
                'pattern': pattern['pattern'],
                'price': current_candle['close'],
                'candle': current_candle
            }
        
        # 4. Technical Exit Conditions
        technical_exit = self._check_technical_exit(current_data)
        
        if technical_exit:
            return {
                'reason': 'technical_exit',
                'condition': technical_exit,
                'price': current_candle['close'],
                'candle': current_candle
            }
        
        return None
    
    def _check_exit_patterns(self, data):
        """
        Check for specific exit patterns from Ross Cameron's strategy.
        
        Parameters:
        -----------
        data: DataFrame
            Price and indicator data
            
        Returns:
        --------
        list: Exit patterns detected
        """
        exit_patterns = []
        
        # Get the last few candles
        if len(data) < 3:
            return []
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 1. Doji candle after a run
        if self._is_doji(current):
            exit_patterns.append({
                'pattern': 'doji',
                'confidence': 70,
                'notes': 'Doji indicates indecision'
            })
        
        # 2. Gravestone Doji
        if self._is_gravestone_doji(current):
            exit_patterns.append({
                'pattern': 'gravestone_doji',
                'confidence': 80,
                'notes': 'Strong reversal signal'
            })
        
        # 3. Shooting Star
        if self._is_shooting_star(current, data):
            exit_patterns.append({
                'pattern': 'shooting_star',
                'confidence': 75,
                'notes': 'Uptrend losing steam'
            })
        
        # 4. Candle Under Candle (first candle to make new low)
        if current['low'] < prev['low']:
            # Check if this is the first lower low after higher highs
            if self._is_first_lower_low(data):
                exit_patterns.append({
                    'pattern': 'candle_under_candle',
                    'confidence': 85,
                    'notes': 'Trend reversal confirmed'
                })
        
        return exit_patterns
    
    def _check_technical_exit(self, data):
        """
        Check technical exit conditions (price below EMAs, MACD negative).
        
        Parameters:
        -----------
        data: DataFrame
            Price and indicator data
            
        Returns:
        --------
        str or None: Exit condition if met
        """
        current = data.iloc[-1]
        
        # Check if price is below 20 EMA
        if 'ema20' in current and current['close'] < current['ema20']:
            return 'price_below_20ema'
        
        # Check if price is below VWAP
        if 'vwap' in current and current['close'] < current['vwap']:
            return 'price_below_vwap'
        
        # Check if MACD turned negative
        if 'macd_line' in current and current['macd_line'] < 0:
            return 'macd_negative'
        
        return None
    
    def _is_doji(self, candle):
        """Check if candle is a doji."""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        return (body_size / total_range) < 0.1
    
    def _is_gravestone_doji(self, candle):
        """Check if candle is a gravestone doji."""
        if not self._is_doji(candle):
            return False
            
        # Long upper wick, close near low
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return upper_wick > (lower_wick * 2)
    
    def _is_shooting_star(self, candle, data):
        """Check if candle is a shooting star."""
        # Small red body at bottom, long upper wick
        if candle['close'] >= candle['open']:  # Not red
            return False
            
        body_size = candle['open'] - candle['close']
        upper_wick = candle['high'] - candle['open']
        total_range = candle['high'] - candle['low']
        
        # Upper wick should be at least 2x body
        if upper_wick < (body_size * 2):
            return False
            
        # Body should be in lower third
        body_position = (candle['close'] - candle['low']) / total_range
        
        return body_position < 0.33
    
    def _is_first_lower_low(self, data):
        """Check if this is the first candle to make a lower low."""
        if len(data) < 5:
            return False
            
        recent = data.iloc[-5:]
        current_low = recent.iloc[-1]['low']
        
        # Check if previous candles had higher lows
        for i in range(len(recent) - 2, 0, -1):
            if recent.iloc[i]['low'] < recent.iloc[i-1]['low']:
                return False  # Already had a lower low
                
        return True
    
    def _check_stop_adjustment(self, trade, current_data):
        """
        Check if stop loss should be adjusted (trailing stop).
        
        Parameters:
        -----------
        trade: dict
            Active trade details
        current_data: DataFrame
            Current price data
            
        Returns:
        --------
        dict or None: Stop adjustment action if needed
        """
        current_candle = current_data.iloc[-1]
        entry_price = trade['entry_price']
        current_price = current_candle['close']
        current_stop = trade['stop_price']
        
        # Only trail if in profit
        if current_price <= entry_price:
            return None
        
        # Calculate profit percentage
        profit_pct = (current_price - entry_price) / entry_price * 100
        
        new_stop = None
        
        # Ross Cameron's trailing approach
        if profit_pct >= 2:  # 2% profit
            # Move stop to breakeven
            new_stop = entry_price
        
        if profit_pct >= 5:  # 5% profit
            # Trail to lock in 2.5% profit
            new_stop = entry_price * 1.025
            
        if profit_pct >= 10:  # 10% profit
            # Trail more aggressively
            new_stop = entry_price * 1.07
        
        # Only update if new stop is higher than current
        if new_stop and new_stop > current_stop:
            trade['stop_price'] = new_stop
            
            # Update in risk manager
            if hasattr(self.risk_manager, 'update_stop'):
                self.risk_manager.update_stop(trade['symbol'], new_stop)
            
            return {
                'action': 'stop_adjusted',
                'symbol': trade['symbol'],
                'old_stop': current_stop,
                'new_stop': new_stop,
                'reason': f'Trailing at {profit_pct:.1f}% profit'
            }
        
        return None
    
    def _execute_trade(self, symbol, pattern, shares):
        """Execute a trade (simulated or real)."""
        entry_price = pattern['entry_price']
        stop_price = pattern['stop_price']
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'pattern': pattern['pattern'],
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'stop_price': stop_price,
            'shares': shares,
            'status': 'open'
        }
        
        # Set target if available
        if 'target_price' in pattern:
            trade['target_price'] = pattern['target_price']
        
        # Add position to risk manager
        self.risk_manager.add_position(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=pattern.get('target_price', entry_price * 1.02),
            shares=shares
        )
        
        self.logger.info(f"Trade executed: {symbol} - Entry: ${entry_price:.2f}, "
                       f"Stop: ${stop_price:.2f}, Shares: {shares}")
        
        return trade
    
    def _exit_trade(self, trade, current_candle, exit_signal):
        """Exit a trade and record results."""
        exit_price = exit_signal['price']
        exit_reason = exit_signal['reason']
        
        # Calculate P&L
        pnl = (exit_price - trade['entry_price']) * trade['shares']
        
        # Update trade record
        trade['exit_time'] = datetime.now()
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'closed'
        
        # Update daily stats
        self.daily_pnl += pnl
        if pnl > 0:
            self.daily_win_count += 1
        else:
            self.daily_loss_count += 1
        
        # Update capital
        self.current_capital += pnl
        
        # Close position in risk manager
        self.risk_manager.close_position(trade['symbol'], exit_price, exit_reason)
        
        # Add to completed trades
        self.completed_trades.append(trade.copy())
        
        self.logger.info(f"Trade closed: {trade['symbol']} - Exit: ${exit_price:.2f}, "
                       f"Reason: {exit_reason}, P&L: ${pnl:.2f}")
        
        return {
            'action': 'trade_closed',
            'symbol': trade['symbol'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl
        }
    
    def _get_stock_data(self, symbol):
        """Get stock data for condition checking."""
        try:
            # Use IB connector to get current data
            current_price = self.ib_connector.get_current_price(symbol)
            contract_details = self.ib_connector.get_contract_details(symbol)
            
            # Get historical data for calculations
            hist_data = self.ib_connector.get_historical_data(
                symbol, duration='2 D', bar_size='1 day'
            )
            
            if hist_data.empty:
                return None
            
            # Calculate metrics
            previous_close = hist_data['close'].iloc[-2] if len(hist_data) > 1 else 0
            day_change_percent = ((current_price - previous_close) / previous_close * 100) if previous_close else 0
            
            # Get volume data
            volume_data = self.ib_connector.get_historical_data(
                symbol, duration='50 D', bar_size='1 day'
            )
            
            avg_volume_50d = volume_data['volume'].mean() if not volume_data.empty else 0
            current_volume = self.ib_connector.get_current_volume(symbol) or 0
            relative_volume = current_volume / avg_volume_50d if avg_volume_50d > 0 else 0
            
            # Get news
            news_provider = self.ib_connector.news_provider if hasattr(self.ib_connector, 'news_provider') else None
            has_news = bool(news_provider and news_provider.get_stock_news(symbol, days=1))
            
            return {
                'current_price': current_price,
                'day_change_percent': day_change_percent,
                'relative_volume': relative_volume,
                'has_news': has_news,
                'shares_float': contract_details.get('shares_float', 0) if contract_details else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def _get_intraday_data(self, symbol):
        """Get intraday data for pattern detection."""
        try:
            # Get 1-minute data for today
            data = self.ib_connector.get_historical_data(
                symbol, duration='1 D', bar_size='1 min', use_rth=True
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_current_data(self, symbol):
        """Get current data including recent candles."""
        try:
            # Get recent 1-minute bars
            data = self.ib_connector.get_historical_data(
                symbol, duration='1 D', bar_size='1 min', use_rth=True
            )
            
            # Add indicators
            if not data.empty:
                data = self._add_indicators(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting current data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_indicators(self, data):
        """Add technical indicators to the data."""
        # Add moving averages
        data = self.moving_averages.add_moving_averages(data)
        
        # Add MACD
        data = self.macd.add_macd(data)
        
        # Calculate VWAP
        data = self._calculate_vwap(data)
        
        return data
    
    def _calculate_vwap(self, data):
        """Calculate VWAP for the data."""
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        data['cumulative_volume'] = data['volume'].cumsum()
        data['cumulative_volume_price'] = (data['volume'] * data['typical_price']).cumsum()
        data['vwap'] = data['cumulative_volume_price'] / data['cumulative_volume']
        
        # Clean up temporary columns
        data.drop(['typical_price', 'cumulative_volume', 'cumulative_volume_price'], axis=1, inplace=True)
        
        return data
    
    def run(self, watch_list, check_interval=60):
        """
        Main trading loop following Ross Cameron's strategy.
        
        Parameters:
        -----------
        watch_list: list
            Stock symbols to monitor
        check_interval: int
            Seconds between checks
        """
        self.logger.info(f"Starting trading bot with watch list: {watch_list}")
        
        if not self.start_trading_session():
            self.logger.error("Failed to start trading session")
            return
        
        try:
            while True:
                # 1. Scan for stocks meeting all 5 conditions
                qualified_stocks = self.scan_stocks(watch_list)
                
                if qualified_stocks:
                    self.logger.info(f"Found {len(qualified_stocks)} stocks meeting all conditions")
                    
                    # 2. Look for entry patterns
                    entry_opportunities = self.find_entry_signals(qualified_stocks)
                    
                    if entry_opportunities:
                        self.logger.info(f"Found {len(entry_opportunities)} entry opportunities")
                        
                        # 3. Execute trades
                        self.execute_trades(entry_opportunities)
                
                # 4. Manage active trades
                actions = self.manage_active_trades()
                
                if actions:
                    self.logger.info(f"Performed {len(actions)} trade management actions")
                
                # 5. Check if we should stop trading for the day
                if self._should_stop_trading():
                    self.logger.info("Stopping trading for the day")
                    break
                
                # Wait for next interval
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
        finally:
            self._cleanup()
    
    def _should_stop_trading(self):
        """Check if we should stop trading for the day."""
        # Stop if daily loss limit reached
        max_loss_reached, _ = self.risk_manager.check_max_daily_loss()
        if max_loss_reached:
            self.logger.warning("Daily loss limit reached")
            return True
        
        # Stop if daily goal reached (optional)
        daily_goal = self.initial_capital * 0.02  # 2% daily goal
        if self.daily_pnl >= daily_goal:
            self.logger.info(f"Daily goal reached: ${self.daily_pnl:.2f}")
            # You might want to continue trading with reduced size
            pass
        
        return False
    
    def _cleanup(self):
        """Clean up resources."""
        # Close all positions
        self.logger.info("Closing all positions")
        
        for symbol in list(self.active_trades.keys()):
            trade = self.active_trades[symbol]
            current_data = self._get_current_data(symbol)
            
            if not current_data.empty:
                current_candle = current_data.iloc[-1]
                exit_signal = {
                    'reason': 'session_end',
                    'price': current_candle['close']
                }
                self._exit_trade(trade, current_candle, exit_signal)
        
        # Disconnect from IB
        if self.ib_connector:
            self.ib_connector.disconnect()
        
        # Print session summary
        self.print_session_summary()
    
    def print_session_summary(self):
        """Print trading session summary."""
        self.logger.info("\n=== Trading Session Summary ===")
        self.logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        self.logger.info(f"Final Capital: ${self.current_capital:.2f}")
        self.logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.info(f"Winning Trades: {self.daily_win_count}")
        self.logger.info(f"Losing Trades: {self.daily_loss_count}")
        
        if self.daily_win_count + self.daily_loss_count > 0:
            win_rate = self.daily_win_count / (self.daily_win_count + self.daily_loss_count) * 100
            self.logger.info(f"Win Rate: {win_rate:.1f}%")
        
        self.logger.info("===============================\n")