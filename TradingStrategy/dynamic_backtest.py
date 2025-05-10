#!/usr/bin/env python
"""
Dynamic Backtesting System

This script runs a backtest using strategies dynamically loaded from the entry folder.
It allows for testing various patterns and strategies without code rewrites.

Usage:
python run_dynamic_backtest.py --symbol GPUS --start-date 2023-05-02 --end-date 2023-05-08 --strategy bull_flag
"""

import os
import sys
import argparse
import importlib
import inspect
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.data.ib_connector import IBConnector
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.trading.risk_manager import RiskManager


class DynamicBacktester:
    """
    A dynamic backtesting system that loads and tests different trading strategies.
    """
    
    def __init__(self, start_date, end_date, initial_capital=10000.0, symbols=None, strategy=None):
        """
        Initialize the dynamic backtester.
        
        Parameters:
        -----------
        start_date: str or datetime
            Start date for backtest
        end_date: str or datetime
            End date for backtest
        initial_capital: float
            Initial capital for backtest
        symbols: list
            List of symbols to backtest
        strategy: str
            Name of the strategy to use
        """
        # Setup logger
        self.logger = setup_logger('dynamic_backtester', log_dir='logs')
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = end_date
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.symbols = symbols or ['GPUS']
        self.strategy_name = strategy
        
        # Initialize components
        self.market_data = MarketDataProvider()
        self.news_data = NewsDataProvider()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Initialize performance tracking
        self.active_trades = {}
        self.completed_trades = []
        
        # Dynamically load strategies
        self.available_strategies = self._discover_strategies()
        self.selected_strategy = self._load_strategy(strategy) if strategy else None
        
        self.logger.info(f"Dynamic Backtester initialized with {len(self.available_strategies)} available strategies")
        if self.selected_strategy:
            self.logger.info(f"Selected strategy: {strategy}")
    
    def _discover_strategies(self):
        """
        Discover available trading strategies from the entry folder.
        
        Returns:
        --------
        dict: Dictionary of strategy name -> strategy class
        """
        strategies = {}
        
        # Path to the entry folder
        entry_path = os.path.join(project_root,'TradingStrategy', 'src', 'entry')
        
        # Ensure the directory exists
        if not os.path.isdir(entry_path):
            self.logger.warning(f"Entry directory not found: {entry_path}")
            return strategies
        
        # Find all Python files in the entry folder
        strategy_files = [f for f in os.listdir(entry_path) 
                          if f.endswith('.py') and f != '__init__.py']
        
        for file_name in strategy_files:
            module_name = file_name[:-3]  # Remove .py extension
            
            try:
                # Import the module
                module_path = f"src.entry.{module_name}"
                module = importlib.import_module(module_path)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Only include classes defined in this module (not imported ones)
                    if obj.__module__ == module_path:
                        # Check if this class has pattern detection methods
                        if any(method.startswith('detect_') for method in dir(obj)):
                            strategies[name.lower()] = obj
                            self.logger.info(f"Discovered strategy: {name} in {file_name}")
            
            except Exception as e:
                self.logger.error(f"Error loading strategy module {module_name}: {e}")
        
        return strategies
    
    def _load_strategy(self, strategy_name):
        """
        Load a specific strategy by name.
        
        Parameters:
        -----------
        strategy_name: str
            Name of the strategy to load
            
        Returns:
        --------
        object: Instantiated strategy object
        """
        if not strategy_name:
            return None
            
        strategy_name = strategy_name.lower()
        
        # Special case for the main candlestick patterns class
        if strategy_name == 'candlestickpatterns':
            from src.entry.candlestick import CandlestickPatterns
            return CandlestickPatterns()
        
        # Check if strategy exists in available strategies
        if strategy_name in self.available_strategies:
            strategy_class = self.available_strategies[strategy_name]
            try:
                # Instantiate the strategy
                return strategy_class()
            except Exception as e:
                self.logger.error(f"Error instantiating strategy {strategy_name}: {e}")
                return None
        
        self.logger.warning(f"Strategy not found: {strategy_name}")
        return None
    
    def list_available_strategies(self):
        """
        List all available strategies.
        
        Returns:
        --------
        list: List of available strategy names
        """
        return list(self.available_strategies.keys())
    
    def set_strategy(self, strategy_name):
        """
        Set the active strategy.
        
        Parameters:
        -----------
        strategy_name: str
            Name of the strategy to use
            
        Returns:
        --------
        bool: Success or failure
        """
        strategy = self._load_strategy(strategy_name)
        if strategy:
            self.strategy_name = strategy_name
            self.selected_strategy = strategy
            self.logger.info(f"Strategy set to: {strategy_name}")
            return True
        
        return False
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for a DataFrame.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            OHLCV data
            
        Returns:
        --------
        pandas.DataFrame: DataFrame with indicators added
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate EMAs
        result['ema9'] = result['close'].ewm(span=9, adjust=False).mean()
        result['ema20'] = result['close'].ewm(span=20, adjust=False).mean()
        result['ema200'] = result['close'].ewm(span=200, adjust=False).mean()
        
        # Calculate MACD
        result['macd_fast'] = result['close'].ewm(span=12, adjust=False).mean()
        result['macd_slow'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd_line'] = result['macd_fast'] - result['macd_slow']
        result['macd_signal'] = result['macd_line'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        
        # Calculate VWAP (daily)
        if 'volume' in result.columns:
            # Calculate typical price
            result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
            
            # Calculate cumulative values
            result['cum_volume'] = result['volume'].cumsum()
            result['cum_volume_price'] = (result['volume'] * result['typical_price']).cumsum()
            
            # Calculate VWAP
            result['vwap'] = result['cum_volume_price'] / result['cum_volume']
            
            # Clean up intermediate columns
            result.drop(['typical_price', 'cum_volume', 'cum_volume_price'], axis=1, inplace=True)
        
        return result
    
    def run_backtest(self):
        """
        Run a backtest using the selected strategy.
        
        Returns:
        --------
        dict: Backtest results
        """
        if not self.selected_strategy:
            self.logger.error("No strategy selected for backtest")
            return None
        
        self.logger.info(f"Starting backtest with strategy: {self.strategy_name}")
        self.logger.info(f"Backtest period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Reset performance metrics
        self.current_capital = self.initial_capital
        self.active_trades = {}
        self.completed_trades = []
        self.risk_manager.reset_daily_metrics()
        
        results_by_symbol = {}
        
        for symbol in self.symbols:
            self.logger.info(f"Backtesting {symbol}")
            
            # Get historical data
            days_diff = (self.end_date - self.start_date).days + 1
            duration = f"{days_diff} D"
            
            df = self._get_historical_data(symbol, duration)
            
            if df.empty:
                self.logger.warning(f"No historical data for {symbol}, skipping")
                continue
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Process data day by day
            symbol_results = self._process_symbol_data(symbol, df)
            results_by_symbol[symbol] = symbol_results
        
        # Combine results
        overall_results = self._generate_backtest_results(results_by_symbol)
        
        return overall_results
    
    def _get_historical_data(self, symbol, duration):
        """
        Get historical data for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        duration: str
            Duration string for IB
            
        Returns:
        --------
        pandas.DataFrame: Historical OHLCV data
        """
        try:
            # Try using IB connector if available
            ib_connector = IBConnector()
            if ib_connector.connect():
                self.logger.info(f"Connected to IB, retrieving data for {symbol}")
                df = ib_connector.get_historical_data(
                    symbol, 
                    duration=duration, 
                    bar_size='1 min', 
                    what_to_show='TRADES', 
                    use_rth=True
                )
                ib_connector.disconnect()
            else:
                self.logger.warning("Failed to connect to IB, using MarketDataProvider")
                df = self.market_data.get_intraday_data(
                    symbol,
                    interval='1m',
                    lookback_days=duration.split()[0]
                )
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_symbol_data(self, symbol, df):
        """
        Process historical data for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        df: pandas.DataFrame
            Historical OHLCV data
            
        Returns:
        --------
        dict: Results for this symbol
        """
        # Reset for this symbol
        symbol_active_trades = {}
        symbol_completed_trades = []
        
        # Process data day by day
        dates = df.index.floor('D').unique()
        
        for date in dates:
            # Get data for this day
            day_data = df[df.index.floor('D') == date]
            if day_data.empty:
                continue
                
            self.logger.info(f"Processing {symbol} for {date.strftime('%Y-%m-%d')}")
            
            # Reset daily metrics
            self.risk_manager.reset_daily_metrics()
            
            # Simulate trading for this day
            self._simulate_trading_day(symbol, day_data, symbol_active_trades, symbol_completed_trades)
        
        # Close any remaining open trades at the end of the backtest
        self._close_all_trades(symbol, df.iloc[-1] if not df.empty else None, 
                              symbol_active_trades, symbol_completed_trades, 'backtest_end')
        
        # Calculate symbol-specific results
        symbol_results = {
            'completed_trades': symbol_completed_trades,
            'trade_count': len(symbol_completed_trades),
            'win_count': sum(1 for t in symbol_completed_trades if t['pnl'] > 0),
            'loss_count': sum(1 for t in symbol_completed_trades if t['pnl'] <= 0),
            'total_pnl': sum(t['pnl'] for t in symbol_completed_trades)
        }
        
        if symbol_results['trade_count'] > 0:
            symbol_results['win_rate'] = symbol_results['win_count'] / symbol_results['trade_count']
        else:
            symbol_results['win_rate'] = 0
            
        self.logger.info(f"Results for {symbol}: {symbol_results['trade_count']} trades, " +
                       f"win rate: {symbol_results['win_rate']:.2f}, " +
                       f"PnL: ${symbol_results['total_pnl']:.2f}")
            
        return symbol_results
    
    def _simulate_trading_day(self, symbol, day_data, active_trades, completed_trades):
        """
        Simulate trading for a single day.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        day_data: pandas.DataFrame
            OHLCV data for a single day
        active_trades: dict
            Dictionary to track active trades
        completed_trades: list
            List to add completed trades
        """
        # Skip if data is too short
        if len(day_data) < 10:
            return
            
        # Check if the stock meets basic criteria (price, volume, etc.)
        if not self._meets_basic_criteria(symbol, day_data):
            self.logger.info(f"{symbol} does not meet basic criteria for trading")
            return
            
        self.logger.info(f"Trading {symbol} for the day with strategy: {self.strategy_name}")
        
        # Loop through each minute of the day
        for i in range(5, len(day_data)):
            # Get current candle and a window of recent data
            current_candle = day_data.iloc[i]
            window = day_data.iloc[max(0, i-20):i+1]
            
            # Manage existing trades first
            self._manage_trades(symbol, current_candle, active_trades, completed_trades)
            
            # Check for new entry signals (if we have capacity)
            if len(active_trades) < self.risk_manager.max_open_positions:
                entry_signal = self._check_for_entry(symbol, window)
                
                if entry_signal:
                    # Execute the trade
                    self._execute_trade(symbol, entry_signal, current_candle, active_trades)
        
        # Close any remaining trades at end of day
        self._close_all_trades(symbol, day_data.iloc[-1], active_trades, completed_trades, 'day_end')
    
    def _meets_basic_criteria(self, symbol, day_data):
        """
        Check if a stock meets basic trading criteria.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        day_data: pandas.DataFrame
            OHLCV data for a single day
            
        Returns:
        --------
        bool: True if criteria are met
        """
        # Get basic data
        if day_data.empty:
            return False
            
        # First/last price of the day
        first_price = day_data['open'].iloc[0]
        last_price = day_data['close'].iloc[-1]
        
        # Check price range ($2-$20)
        if not (2.0 <= last_price <= 20.0):
            return False
            
        # For backtesting purposes, assume these criteria are met:
        # - Up at least 10% (would need previous day close)
        # - 5x relative volume
        # - Low float
        # - Has news
        
        return True
    
    def _check_for_entry(self, symbol, window):
        """
        Check for entry signals based on the selected strategy.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        window: pandas.DataFrame
            Window of recent OHLCV data
            
        Returns:
        --------
        dict or None: Entry signal if found
        """
        if not self.selected_strategy:
            return None
            
        # Get detection methods from the strategy
        detection_methods = [method for method in dir(self.selected_strategy) 
                           if method.startswith('detect_') and callable(getattr(self.selected_strategy, method))]
        
        # Check each detection method
        for method_name in detection_methods:
            try:
                method = getattr(self.selected_strategy, method_name)
                pattern = method(window)
                
                if pattern and isinstance(pattern, dict) and pattern.get('is_valid', False):
                    pattern_name = pattern.get('pattern', method_name.replace('detect_', ''))
                    self.logger.info(f"Entry signal: {pattern_name} for {symbol}")
                    return pattern
            except Exception as e:
                self.logger.error(f"Error checking {method_name} for {symbol}: {e}")
        
        return None
    
    def _execute_trade(self, symbol, entry_signal, current_candle, active_trades):
        """
        Execute a simulated trade.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        entry_signal: dict
            Entry signal from strategy
        current_candle: pandas.Series
            Current price data
        active_trades: dict
            Dictionary to track active trades
        """
        # Get trade parameters
        entry_price = entry_signal['entry_price']
        stop_price = entry_signal['stop_price']
        pattern = entry_signal['pattern']
        
        # Ensure stop is not too tight (at least 2% from entry)
        min_stop_distance = entry_price * 0.02
        if entry_price - stop_price < min_stop_distance:
            stop_price = entry_price - min_stop_distance
            
        # Calculate target based on profit-loss ratio
        risk_per_share = entry_price - stop_price
        target_price = entry_price + (risk_per_share * self.risk_manager.profit_loss_ratio)
        
        # Calculate position size
        shares = self.risk_manager.calculate_position_size(entry_price, stop_price)
        
        if shares <= 0:
            self.logger.warning(f"Invalid position size for {symbol}")
            return
            
        # Generate unique trade ID
        trade_id = f"{symbol}_{current_candle.name.strftime('%Y%m%d%H%M%S')}_{len(active_trades)}"
        
        # Create trade record
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'pattern': pattern,
            'entry_time': current_candle.name,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'status': 'open'
        }
        
        # Add to active trades
        active_trades[trade_id] = trade
        
        # Add position to risk manager
        self.risk_manager.add_position(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            shares=shares
        )
        
        self.logger.info(f"Trade executed: {symbol} ({pattern}) - Entry: ${entry_price:.2f}, " +
                       f"Stop: ${stop_price:.2f}, Target: ${target_price:.2f}, Shares: {shares}")
    
    def _manage_trades(self, symbol, current_candle, active_trades, completed_trades):
        """
        Manage active trades.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        current_candle: pandas.Series
            Current price data
        active_trades: dict
            Dictionary of active trades
        completed_trades: list
            List to add completed trades
        """
        # Get current prices
        high = current_candle['high']
        low = current_candle['low']
        close = current_candle['close']
        
        # Check trades for this symbol
        trades_to_close = []
        
        for trade_id, trade in active_trades.items():
            if trade['symbol'] != symbol:
                continue
                
            # Check if stop loss hit
            if low <= trade['stop_price']:
                self._close_trade(trade, current_candle, trade['stop_price'], 'stop_loss', 
                                active_trades, completed_trades)
                trades_to_close.append(trade_id)
                continue
                
            # Check if target reached
            if high >= trade['target_price']:
                self._close_trade(trade, current_candle, trade['target_price'], 'target_reached', 
                                active_trades, completed_trades)
                trades_to_close.append(trade_id)
                continue
                
            # Check for trailing stop adjustments (only for trades in profit)
            if close > trade['entry_price']:
                self._update_trailing_stop(trade, current_candle)
        
        # Remove closed trades
        for trade_id in trades_to_close:
            if trade_id in active_trades:
                del active_trades[trade_id]
    
    def _update_trailing_stop(self, trade, current_candle):
        """
        Update trailing stop for a trade in profit.
        
        Parameters:
        -----------
        trade: dict
            Trade details
        current_candle: pandas.Series
            Current price data
        """
        entry_price = trade['entry_price']
        current_price = current_candle['close']
        target_price = trade['target_price']
        original_stop = trade.get('original_stop', trade['stop_price'])
        
        # Only update if we're in profit
        if current_price <= entry_price:
            return
            
        # Calculate progress to target
        progress_to_target = (current_price - entry_price) / (target_price - entry_price)
        
        # Different trailing strategies based on progress
        if progress_to_target >= 0.8:  # 80% of the way to target
            # Trail to 75% of gains (aggressive trailing)
            new_stop = entry_price + ((current_price - entry_price) * 0.75)
        elif progress_to_target >= 0.5:  # 50-80% of the way
            # Trail to breakeven plus 50% of gains
            new_stop = entry_price + ((current_price - entry_price) * 0.5)
        elif progress_to_target >= 0.3:  # 30-50% of the way
            # Trail to breakeven
            new_stop = entry_price
        else:
            # Keep original stop
            return
            
        # Only update if new stop is higher
        if new_stop > trade['stop_price']:
            # Save original stop if not saved yet
            if 'original_stop' not in trade:
                trade['original_stop'] = trade['stop_price']
                
            # Update stop
            old_stop = trade['stop_price']
            trade['stop_price'] = new_stop
            
            self.logger.info(f"Trailing stop adjusted: {trade['symbol']} - " +
                           f"${old_stop:.2f} -> ${new_stop:.2f} " +
                           f"({progress_to_target:.1%} to target)")
            
            # Update in risk manager too
            if trade['symbol'] in self.risk_manager.open_positions:
                self.risk_manager.open_positions[trade['symbol']]['stop_price'] = new_stop
    
    def _close_trade(self, trade, current_candle, exit_price, exit_reason, active_trades, completed_trades):
        """
        Close a trade and update metrics.
        
        Parameters:
        -----------
        trade: dict
            Trade details
        current_candle: pandas.Series
            Current price data
        exit_price: float
            Exit price
        exit_reason: str
            Reason for exit
        active_trades: dict
            Dictionary of active trades
        completed_trades: list
            List to add completed trades
        """
        # Calculate P&L
        shares = trade['shares']
        entry_price = trade['entry_price']
        pnl = (exit_price - entry_price) * shares
        
        # Update trade record
        trade_result = trade.copy()
        trade_result['exit_time'] = current_candle.name
        trade_result['exit_price'] = exit_price
        trade_result['exit_reason'] = exit_reason
        trade_result['pnl'] = pnl
        trade_result['status'] = 'closed'
        
        # Calculate duration
        if hasattr(current_candle.name, 'total_seconds'):
            trade_result['duration_seconds'] = (current_candle.name - trade['entry_time']).total_seconds()
        
        # Add to completed trades
        completed_trades.append(trade_result)
        
        # Update account value
        self.current_capital += pnl
        
        # Close position in risk manager
        symbol = trade['symbol']
        if symbol in self.risk_manager.open_positions:
            self.risk_manager.close_position(symbol, exit_price, exit_reason)
        
        action = "Win" if pnl > 0 else "Loss"
        self.logger.info(f"{action}: {symbol} ({exit_reason}) at ${exit_price:.2f} - PnL: ${pnl:.2f}")
    
    def _close_all_trades(self, symbol, last_candle, active_trades, completed_trades, reason):
        """
        Close all open trades for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        last_candle: pandas.Series
            Last available price data
        active_trades: dict
            Dictionary of active trades
        completed_trades: list
            List to add completed trades
        reason: str
            Reason for closing trades
        """
        trades_to_close = [trade_id for trade_id, trade in active_trades.items() 
                         if trade['symbol'] == symbol]
        
        for trade_id in trades_to_close:
            trade = active_trades[trade_id]
            exit_price = last_candle['close'] if last_candle is not None else trade['entry_price']
            
            self._close_trade(trade, last_candle, exit_price, reason, active_trades, completed_trades)
            
            if trade_id in active_trades:
                del active_trades[trade_id]
    
    def _generate_backtest_results(self, results_by_symbol):
        """
        Generate overall backtest results.
        
        Parameters:
        -----------
        results_by_symbol: dict
            Results for each symbol
            
        Returns:
        --------
        dict: Overall backtest results
        """
        # Combine all completed trades
        all_trades = []
        for symbol, results in results_by_symbol.items():
            all_trades.extend(results['completed_trades'])
        
        # Sort trades by entry time
        all_trades.sort(key=lambda x: x['entry_time'])
        
        # Calculate overall metrics
        total_trades = len(all_trades)
        winning_trades = sum(1 for t in all_trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        total_profit = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
        total_loss = sum(t['pnl'] for t in all_trades if t['pnl'] <= 0)
        total_pnl = total_profit + total_loss
        
        # Calculate win rate and profit factor
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate metrics by pattern
        pattern_stats = {}
        
        for trade in all_trades:
            pattern = trade['pattern']
            
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0.0,
                    'loss': 0.0,
                    'net_pnl': 0.0
                }
            
            pattern_stats[pattern]['count'] += 1
            
            if trade['pnl'] > 0:
                pattern_stats[pattern]['wins'] += 1
                pattern_stats[pattern]['profit'] += trade['pnl']
            else:
                pattern_stats[pattern]['losses'] += 1
                pattern_stats[pattern]['loss'] += trade['pnl']
                
            pattern_stats[pattern]['net_pnl'] += trade['pnl']
        
        # Calculate pattern win rates and profit factors
        for pattern, stats in pattern_stats.items():
            stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            stats['profit_factor'] = abs(stats['profit'] / stats['loss']) if stats['loss'] < 0 else float('inf')
        
        # Generate results dictionary
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_pnl': total_pnl,
            'return_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'pattern_stats': pattern_stats,
            'symbols_tested': list(results_by_symbol.keys()),
            'strategy_used': self.strategy_name,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'trades': all_trades
        }
        
        return results
    
    def display_results(self, results):
        """
        Display backtest results in a readable format.
        
        Parameters:
        -----------
        results: dict
            Backtest results
        """
        if not results:
            self.logger.warning("No results to display")
            return
            
        print("\n========== BACKTEST RESULTS ==========")
        print(f"Strategy: {results['strategy_used']}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Symbols: {', '.join(results['symbols_tested'])}")
        print("\n---------- Performance -----------")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print("\n---------- Trade Statistics ----------")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.2%})")
        print(f"Losing Trades: {results['losing_trades']} ({1-results['win_rate']:.2%})")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        print("\n---------- Pattern Performance ----------")
        for pattern, stats in results['pattern_stats'].items():
            print(f"\n{pattern.upper()}:")
            print(f"  Trades: {stats['count']}")
            print(f"  Win Rate: {stats['win_rate']:.2%}")
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
            print(f"  Net P&L: ${stats['net_pnl']:.2f}")
        
        print("\n======================================\n")
        
        # Return the results for potential further processing
        return results