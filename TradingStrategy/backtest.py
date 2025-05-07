"""
Backtesting Script for Ross Cameron's Day Trading Strategy

This script runs the day trading strategy on historical data to evaluate performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import time
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.scanning.scanner import StockScanner
from src.scanning.condition_tracker import ConditionTracker
from src.trading.risk_manager import RiskManager
from src.trading.trade_manager import TradeManager
from src.utils.logger import setup_logger

# Import the mock data providers
from mock_data_providers import MockMarketDataProvider, MockNewsDataProvider

# Custom JSON Encoder for handling datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)


class BacktestEngine:
    """Backtesting engine for the day trading strategy."""
    
    def __init__(self, config_file='config.yaml', start_date=None, end_date=None, 
                initial_capital=10000.0):
        """
        Initialize the backtesting engine.
        
        Parameters:
        -----------
        config_file: str
            Path to configuration file
        start_date: str
            Start date for backtest in YYYY-MM-DD format
        end_date: str
            End date for backtest in YYYY-MM-DD format
        initial_capital: float
            Initial capital for backtesting
        """
        self.config_file = config_file
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=30)
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        self.initial_capital = initial_capital
        
        # Setup logger
        self.logger = setup_logger('backtest', log_dir='logs', console_level=logging.INFO)
        
        # Initialize components
        self.market_data = None
        self.news_data = None
        self.scanner = None
        self.condition_tracker = None
        self.risk_manager = None
        self.trade_manager = None
        
        # Results storage
        self.daily_results = {}
        self.trade_history = []
        self.equity_curve = []
        
        self.logger.info(f"Backtest initialized from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def setup_components(self):
        """Initialize strategy components."""
        self.logger.info("Setting up strategy components")
        
        # Initialize data providers with historical data capabilities
        self.market_data = MockMarketDataProvider(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.news_data = MockNewsDataProvider(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Initialize scanner and condition tracker
        self.scanner = StockScanner(self.market_data, self.news_data)
        self.condition_tracker = ConditionTracker(self.market_data, self.news_data)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=1.0,  # Risk 1% per trade as per Cameron's rule
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,  # 2:1 profit-to-loss ratio as per Cameron's rule
            max_open_positions=3
        )
        
        # Initialize trade manager
        self.trade_manager = TradeManager(
            market_data_provider=self.market_data,
            scanner=self.scanner,
            risk_manager=self.risk_manager,
            condition_tracker=self.condition_tracker,
            broker_api=None  # No broker for backtesting
        )
        
        # Set trading parameters - amended to better match Ross Cameron's criteria
        self.trade_manager.min_price = 2.0  # Changed from 1.0 to 2.0 to match Cameron's criteria
        self.trade_manager.max_price = 20.0
        self.trade_manager.min_gap_pct = 10.0  # Changed back to 10.0 to match Cameron's criteria
        self.trade_manager.min_rel_volume = 5.0  # Changed back to 5.0 to match Cameron's criteria
        self.trade_manager.max_float = 10_000_000  # Changed back to 10M to match Cameron's criteria
        
        # Enable trading
        self.trade_manager.is_trading_enabled = True
        self.logger.info("Trading enabled for simulation")
        
        # Start trading session to initialize metrics
        self.trade_manager.start_trading_session()
        
        # Override the is_market_healthy method for the entire simulation
        self.condition_tracker.is_market_healthy = lambda: True
        
        self.logger.info("Strategy components initialized")
    
    def run_backtest(self):
        """Run the backtest through the specified date range."""
        self.logger.info("Starting backtest")
        
        # Setup components
        self.setup_components()
        
        # Generate trading days between start and end date (excluding weekends)
        current_date = self.start_date
        total_days = (self.end_date - self.start_date).days + 1
        days_processed = 0
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                current_date += timedelta(days=1)
                continue
                
            # Print progress
            days_processed += 1
            self.logger.info(f"Processing trading day: {current_date.strftime('%Y-%m-%d')} [{days_processed}/{total_days}]")
            
            # Set current date for data providers
            self.market_data.set_current_date(current_date)
            self.news_data.set_current_date(current_date)
            
            # Reset daily metrics
            self.risk_manager.reset_daily_metrics()
            self.trade_manager.daily_stats = {
                'trades_taken': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'gross_pnl': 0.0
            }
            
            # Run trading day simulation
            self.run_trading_day(current_date)
            
            # Store daily results
            self.daily_results[current_date.strftime('%Y-%m-%d')] = {
                'trades_taken': self.trade_manager.daily_stats['trades_taken'],
                'winning_trades': self.trade_manager.daily_stats['winning_trades'],
                'losing_trades': self.trade_manager.daily_stats['losing_trades'],
                'win_rate': (self.trade_manager.daily_stats['winning_trades'] / 
                          self.trade_manager.daily_stats['trades_taken'] 
                          if self.trade_manager.daily_stats['trades_taken'] > 0 else 0),
                'total_profit': self.trade_manager.daily_stats['total_profit'],
                'total_loss': self.trade_manager.daily_stats['total_loss'],
                'net_pnl': self.trade_manager.daily_stats['gross_pnl'],
                'equity': self.risk_manager.current_capital
            }
            
            # Update equity curve
            self.equity_curve.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'equity': self.risk_manager.current_capital
            })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        self.logger.info("Backtest completed")
        
        # Generate and return results
        return self.generate_results()
    
    def run_trading_day(self, current_date):
        """
        Simulate a trading day with more realistic entry conditions.
        
        Parameters:
        -----------
        current_date: datetime
            The date to simulate
        """
        # Force market conditions to be favorable for trading
        self.condition_tracker.market_open = True
        self.condition_tracker.pre_market = False
        self.condition_tracker.post_market = False
        self.condition_tracker.strong_market = True
        
        # Override the is_market_healthy method for simulation
        self.condition_tracker.is_market_healthy = lambda: True
        
        # Morning scan for opportunities (9:30 AM - 10:30 AM)
        morning_time = current_date.replace(hour=9, minute=30)
        self.market_data.set_current_datetime(morning_time)
        self.news_data.set_current_datetime(morning_time)
        
        # DEBUG: Get all tradable stocks and print details
        all_stocks = self.market_data.get_tradable_stocks()
        self.logger.info(f"Found {len(all_stocks)} tradable stocks")
        for stock in all_stocks[:5]:  # Print first 5 for debugging
            self.logger.info(f"Stock {stock.symbol}: price=${stock.current_price:.2f}, " +
                          f"gap={stock.gap_percent:.2f}%, volume={stock.relative_volume:.2f}x, " +
                          f"has_bull_flag={stock.has_bull_flag}, has_news={stock.has_news}")
        
        # Scan for stocks that meet the initial criteria
        filtered_stocks = self.scanner.scan_for_momentum_stocks(
            min_price=self.trade_manager.min_price,
            max_price=self.trade_manager.max_price,
            min_gap_pct=self.trade_manager.min_gap_pct,
            min_rel_volume=self.trade_manager.min_rel_volume,
            max_float=self.trade_manager.max_float
        )
        self.logger.info(f"Found {len(filtered_stocks)} stocks meeting initial criteria")
        
        # Initialize watchlist of potential trades
        watchlist = {}
        for stock in filtered_stocks:
            watchlist[stock.symbol] = {
                'stock': stock,
                'entry_found': False,
                'entry_price': None,
                'stop_price': None,
                'target_price': None,
                'pattern': self._determine_primary_pattern(stock)
            }
        
        # Force condition tracker to track these stocks
        self.condition_tracker.tracked_stocks = {}
        self.condition_tracker.bull_flags = {}
        self.condition_tracker.micro_pullbacks = {}
        self.condition_tracker.new_high_breakouts = {}
        
        for symbol, watch_data in watchlist.items():
            stock = watch_data['stock']
            pattern = watch_data['pattern']
            
            self.condition_tracker.tracked_stocks[symbol] = pattern
            
            if pattern == 'bull_flag':
                stock.has_bull_flag = True
                self.condition_tracker.bull_flags[symbol] = stock
            elif pattern == 'micro_pullback':
                stock.has_micro_pullback = True
                self.condition_tracker.micro_pullbacks[symbol] = stock
            elif pattern == 'new_high_breakout':
                stock.has_new_high_breakout = True
                self.condition_tracker.new_high_breakouts[symbol] = stock
        
        # Clear trade manager active trades at the start of each day
        self.trade_manager.active_trades = {}
        
        # Simulate minute-by-minute market data through the trading day
        for hour in range(9, 16):
            # Skip lunch hour when trading typically slows
            if hour == 12:
                continue
                
            for minute in range(0, 60, 5):  # Check every 5 minutes for performance
                if len(watchlist) == 0 or len(self.trade_manager.active_trades) >= self.risk_manager.max_open_positions:
                    break
                    
                # Update current time
                current_time = current_date.replace(hour=hour, minute=minute)
                self.market_data.set_current_datetime(current_time)
                
                # Skip if past traditional trading time window (3:30 PM)
                if hour >= 15 and minute >= 30:
                    break
                
                # For each stock in our watchlist, check for entry signals
                for symbol, watch_data in list(watchlist.items()):
                    if watch_data['entry_found']:
                        continue  # Already found entry for this stock
                        
                    stock = watch_data['stock']
                    pattern = watch_data['pattern']
                    
                    # Get latest intraday data
                    intraday_data = self.market_data.get_intraday_data(
                        symbol, interval='1m', lookback_days=1
                    )
                    
                    if intraday_data.empty:
                        continue
                        
                    # Update stock with latest data
                    stock.set_price_history(intraday_data)
                    
                    # Check for entry signal based on the pattern
                    entry_signal = self._check_for_entry_signal(stock, pattern)
                    
                    if entry_signal:
                        # Entry signal found, get trade parameters
                        trade_params = self._get_trade_parameters(stock, pattern, entry_signal)
                        
                        if trade_params:
                            # Log trade details for debugging
                            self.logger.info(f"Entry signal detected for {symbol} ({pattern}): " +
                                          f"Entry=${trade_params['entry_price']:.2f}, " +
                                          f"Stop=${trade_params['stop_price']:.2f}, " +
                                          f"Target=${trade_params['target_price']:.2f}")
                            
                            # Execute trade
                            executed_trade = self.trade_manager.execute_trade(trade_params)
                            
                            if executed_trade:
                                self.logger.info(f"Executed trade for {symbol}: " +
                                              f"{executed_trade['executed_shares']} shares at " +
                                              f"${executed_trade['executed_price']:.2f}")
                                self.trade_history.append(executed_trade)
                                
                                # Mark entry as found
                                watch_data['entry_found'] = True
                                watch_data['entry_price'] = executed_trade['executed_price']
                                watch_data['stop_price'] = executed_trade['stop_price']
                                watch_data['target_price'] = executed_trade['target_price']
                                
                                # Remove from watchlist if max positions reached
                                if len(self.trade_manager.active_trades) >= self.risk_manager.max_open_positions:
                                    break
                            else:
                                self.logger.info(f"Failed to execute trade for {symbol}")
                
                # Manage active trades
                if self.trade_manager.active_trades:
                    actions = self.trade_manager.manage_active_trades()
                    
                    if actions:
                        for action in actions:
                            self.logger.info(f"Trade management action: {action['symbol']} - " +
                                          f"{action['action']} - {action['reason']}")
        
        # Close all positions at end of day (3:55 PM)
        closing_time = current_date.replace(hour=15, minute=55)
        self.market_data.set_current_datetime(closing_time)
        
        # Get list of active trades
        active_trades = list(self.trade_manager.active_trades.keys())
        
        # Close each position
        for symbol in active_trades:
            current_price = self.market_data.get_current_price(symbol)
            if current_price:
                try:
                    # Check if the symbol exists in active_trades before exiting
                    if symbol in self.trade_manager.active_trades:
                        self.trade_manager._exit_trade(symbol, current_price, 'end_of_day')
                        self.logger.info(f"Closed position at end of day: {symbol} at ${current_price:.2f}")
                except Exception as e:
                    self.logger.error(f"Error closing position for {symbol}: {str(e)}")
    
    def _determine_primary_pattern(self, stock):
        """
        Determine the primary pattern a stock is displaying.
        
        Parameters:
        -----------
        stock: Stock
            The stock to evaluate
            
        Returns:
        --------
        str: Pattern name ('bull_flag', 'micro_pullback', 'new_high_breakout')
        """
        # Prioritize patterns based on Ross Cameron's preferences
        if stock.has_bull_flag:
            return 'bull_flag'
        elif stock.has_new_high_breakout:
            return 'new_high_breakout'
        elif stock.has_micro_pullback:
            return 'micro_pullback'
        else:
            # Default to bull flag if no specific pattern detected
            return 'bull_flag'
    
    def _check_for_entry_signal(self, stock, pattern):
        """
        Check for specific entry signals based on candlestick patterns.
        
        Parameters:
        -----------
        stock: Stock
            Stock object with updated price history
        pattern: str
            Pattern type to check for
            
        Returns:
        --------
        dict or None: Entry signal details if found, None otherwise
        """
        if pattern == 'bull_flag':
            return self._check_bull_flag_entry(stock)
        elif pattern == 'micro_pullback':
            return self._check_micro_pullback_entry(stock)
        elif pattern == 'new_high_breakout':
            return self._check_new_high_entry(stock)
        
        return None
    
    def _check_bull_flag_entry(self, stock):
        """
        Check for bull flag breakout entry pattern.
        
        Parameters:
        -----------
        stock: Stock
            Stock object with price history
            
        Returns:
        --------
        dict or None: Entry signal details if found, None otherwise
        """
        if stock.price_history is None or len(stock.price_history) < 5:
            return None
        
        # Get recent price data
        recent_data = stock.price_history.iloc[-5:]
        
        # Need at least 5 candles to determine flag pattern
        if len(recent_data) < 5:
            return None
        
        # Calculate flag high (resistance) - exclude most recent candle
        flag_high = recent_data['high'].iloc[:-1].max()
        
        # Get the latest candle
        latest_candle = recent_data.iloc[-1]
        
        # Entry criteria for bull flag:
        # 1. Price breaks above flag resistance 
        # 2. Candle is bullish (close > open)
        # 3. Above average volume on breakout if volume data available
        price_breakout = latest_candle['close'] > flag_high
        bullish_candle = latest_candle['close'] > latest_candle['open']
        
        strong_volume = True  # Default if no volume data
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].iloc[:-1].mean()
            strong_volume = latest_candle['volume'] > avg_volume * 1.2
        
        # Only return entry if all criteria met
        if price_breakout and bullish_candle and strong_volume:
            return {
                'entry_price': latest_candle['close'],
                'flag_high': flag_high,
                'candle_low': latest_candle['low'],
                'volume': latest_candle.get('volume', 0)
            }
        
        return None
    
    def _check_micro_pullback_entry(self, stock):
        """
        Check for micro pullback entry pattern.
        
        Parameters:
        -----------
        stock: Stock
            Stock object with price history
            
        Returns:
        --------
        dict or None: Entry signal details if found, None otherwise
        """
        if stock.price_history is None or len(stock.price_history) < 4:
            return None
        
        # Get recent price data
        recent_data = stock.price_history.iloc[-4:]
        
        # We need at least 4 candles (3 for uptrend, 1 for pullback)
        if len(recent_data) < 4:
            return None
        
        # Check for uptrend in the first 3 candles
        uptrend_candles = recent_data.iloc[:-1]
        uptrend = uptrend_candles['close'].iloc[-1] > uptrend_candles['close'].iloc[0]
        
        if not uptrend:
            return None
        
        # Get pullback candle and the candle after it
        pullback_candle = recent_data.iloc[-2]
        latest_candle = recent_data.iloc[-1]
        
        # Criteria for micro pullback entry:
        # 1. Pullback candle is red or has a long lower wick
        # 2. Latest candle breaks above the pullback candle's high
        # 3. Latest candle is bullish
        is_pullback = (pullback_candle['close'] < pullback_candle['open'] or 
                    (pullback_candle['low'] < min(pullback_candle['open'], pullback_candle['close']) * 0.995))
        
        breaks_above = latest_candle['high'] > pullback_candle['high']
        bullish_candle = latest_candle['close'] > latest_candle['open']
        
        # Only return entry if all criteria met
        if is_pullback and breaks_above and bullish_candle:
            return {
                'entry_price': latest_candle['close'],
                'pullback_low': pullback_candle['low'],
                'candle_low': latest_candle['low'],
                'volume': latest_candle.get('volume', 0)
            }
        
        return None
    
    def _check_new_high_entry(self, stock):
        """
        Check for "first candle to make a new high" entry pattern.
        
        Parameters:
        -----------
        stock: Stock
            Stock object with price history
            
        Returns:
        --------
        dict or None: Entry signal details if found, None otherwise
        """
        if stock.price_history is None or len(stock.price_history) < 6:
            return None
        
        # Get recent price data
        recent_data = stock.price_history.iloc[-6:]
        
        # Need at least 6 candles (5 for lookback, 1 for new high)
        if len(recent_data) < 6:
            return None
        
        # Calculate the previous high in the lookback period
        previous_high = recent_data['high'].iloc[:-1].max()
        
        # Get the latest candle
        latest_candle = recent_data.iloc[-1]
        
        # Entry criteria for new high breakout:
        # 1. Current candle breaks above the previous high
        # 2. Current candle is bullish
        # 3. Strong close near the high of the candle
        breaks_above = latest_candle['high'] > previous_high
        bullish_candle = latest_candle['close'] > latest_candle['open']
        strong_close = (latest_candle['close'] > (latest_candle['high'] - 
                                              (latest_candle['high'] - latest_candle['low']) * 0.25))
        
        # Only return entry if all criteria met
        if breaks_above and bullish_candle and strong_close:
            return {
                'entry_price': latest_candle['close'],
                'previous_high': previous_high,
                'candle_low': latest_candle['low'],
                'volume': latest_candle.get('volume', 0)
            }
        
        return None
    
    def _get_trade_parameters(self, stock, pattern, entry_signal):
        """
        Calculate trade parameters based on the entry signal.
        
        Parameters:
        -----------
        stock: Stock
            Stock object
        pattern: str
            Pattern type
        entry_signal: dict
            Entry signal details
            
        Returns:
        --------
        dict: Trade parameters
        """
        entry_price = entry_signal['entry_price']
        
        # Calculate stop loss based on pattern type
        if pattern == 'bull_flag':
            # For bull flag, place stop below the low of the breakout candle
            # or below the flag low, whichever is higher (tighter)
            candle_low = entry_signal['candle_low']
            stop_buffer = entry_price * 0.005  # 0.5% buffer
            stop_price = candle_low - stop_buffer
        elif pattern == 'micro_pullback':
            # For micro pullback, place stop below the pullback low
            pullback_low = entry_signal['pullback_low']
            stop_buffer = entry_price * 0.005  # 0.5% buffer
            stop_price = pullback_low - stop_buffer
        elif pattern == 'new_high_breakout':
            # For new high breakout, place stop below the breakout candle's low
            candle_low = entry_signal['candle_low']
            stop_buffer = entry_price * 0.005  # 0.5% buffer
            stop_price = candle_low - stop_buffer
        else:
            # Default stop calculation (2% below entry)
            stop_price = entry_price * 0.98
        
        # Ensure stop loss is not too tight (at least 1.5% away)
        min_stop_distance = entry_price * 0.015
        if entry_price - stop_price < min_stop_distance:
            stop_price = entry_price - min_stop_distance
        
        # Calculate target based on risk-reward ratio (2:1)
        risk_per_share = entry_price - stop_price
        target_price = entry_price + (risk_per_share * 2.0)
        
        # Calculate position size
        shares = self.risk_manager.calculate_position_size(entry_price, stop_price, stock)
        
        # Create trade parameters
        trade_params = {
            'symbol': stock.symbol,
            'pattern': pattern,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'risk_per_share': risk_per_share,
            'reward_per_share': target_price - entry_price,
            'dollar_risk': risk_per_share * shares,
            'dollar_reward': (target_price - entry_price) * shares,
            'profit_loss_ratio': (target_price - entry_price) / risk_per_share,
            'timestamp': datetime.now()
        }
        
        return trade_params
    
    def generate_results(self):
        """Generate and return backtest results."""
        self.logger.info("Generating backtest results")
        
        # Calculate overall performance metrics
        total_trades = sum(day['trades_taken'] for day in self.daily_results.values())
        winning_trades = sum(day['winning_trades'] for day in self.daily_results.values())
        losing_trades = sum(day['losing_trades'] for day in self.daily_results.values())
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(day['total_profit'] for day in self.daily_results.values())
        total_loss = abs(sum(day['total_loss'] for day in self.daily_results.values()))
        net_pnl = sum(day['net_pnl'] for day in self.daily_results.values())
        
        # Calculate return metrics
        initial_equity = self.initial_capital
        final_equity = self.risk_manager.current_capital
        
        total_return = ((final_equity / initial_equity) - 1) * 100
        daily_returns = [
            (self.daily_results[date]['net_pnl'] / 
             (initial_equity + sum(self.daily_results[d]['net_pnl'] for d in self.daily_results.keys() if d < date)))
            for date in self.daily_results.keys()
        ]
        
        # Only include non-zero daily returns
        non_zero_returns = [r for r in daily_returns if r != 0]
        
        # Calculate annualized metrics
        trading_days = len(self.daily_results)
        if trading_days > 0:
            annualized_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            
            excess_returns = [r - daily_risk_free for r in non_zero_returns]
            
            if len(excess_returns) > 1:
                sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate drawdown
            equity_values = [initial_equity] + [day['equity'] for day in self.daily_results.values()]
            max_drawdown = 0
            peak = equity_values[0]
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            max_drawdown *= 100  # Convert to percentage
        else:
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Compile results
        results = {
            'summary': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'initial_capital': initial_equity,
                'final_capital': final_equity,
                'total_return_pct': total_return,
                'annualized_return_pct': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_pnl': net_pnl,
                'profit_factor': profit_factor
            },
            'daily_results': self.daily_results,
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history
        }
        
        # Save results to JSON file
        results_file = f"backtest_results_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.json"
        with open(os.path.join('logs', results_file), 'w') as f:
            # Use custom DateTimeEncoder to handle datetime objects
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
        
        self.logger.info(f"Results saved to logs/{results_file}")
        
        # Generate plots if there are trades
        if total_trades > 0:
            self._generate_performance_plots()
        
        return results
    
    def _generate_performance_plots(self):
        """Generate performance plots from the backtest results."""
        # Create equity curve plot
        plt.figure(figsize=(12, 6))
        dates = [item['date'] for item in self.equity_curve]
        equity = [item['equity'] for item in self.equity_curve]
        plt.plot(dates, equity, marker='o')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('logs/equity_curve.png')
        
        # Create daily P&L plot
        plt.figure(figsize=(12, 6))
        dates = list(self.daily_results.keys())
        pnl = [day['net_pnl'] for day in self.daily_results.values()]
        colors = ['green' if x >= 0 else 'red' for x in pnl]
        plt.bar(dates, pnl, color=colors)
        plt.title('Daily P&L')
        plt.xlabel('Date')
        plt.ylabel('P&L ($)')
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('logs/daily_pnl.png')
        
        # Create win rate plot
        plt.figure(figsize=(10, 6))
        trades_taken = [day['trades_taken'] for day in self.daily_results.values()]
        win_rates = [day['win_rate'] * 100 for day in self.daily_results.values()]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Trades Taken', color=color)
        ax1.bar(dates, trades_taken, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Win Rate (%)', color=color)
        ax2.plot(dates, win_rates, color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Daily Trades and Win Rate')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        fig.tight_layout()
        plt.savefig('logs/win_rate.png')
        
        # Add a pattern distribution plot
        plt.figure(figsize=(10, 6))
        patterns = [trade['pattern'] for trade in self.trade_history]
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        if pattern_counts:
            labels = pattern_counts.keys()
            sizes = pattern_counts.values()
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                  shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Trade Pattern Distribution')
            plt.tight_layout()
            plt.savefig('logs/pattern_distribution.png')
            
        # Add entry timing distribution
        plt.figure(figsize=(12, 6))
        entry_times = []
        for trade in self.trade_history:
            if 'execution_time' in trade:
                try:
                    time_obj = datetime.strptime(trade['execution_time'], '%Y-%m-%d %H:%M:%S')
                    entry_times.append(time_obj.hour * 60 + time_obj.minute)
                except (ValueError, TypeError):
                    continue
        
        if entry_times:
            plt.hist(entry_times, bins=range(9*60, 16*60, 30), alpha=0.75, edgecolor='black')
            
            # Set x-ticks to show hours
            plt.xticks([h*60 for h in range(9, 17)], 
                     [f'{h}:00' for h in range(9, 17)])
            
            plt.title('Trade Entry Time Distribution')
            plt.xlabel('Time of Day')
            plt.ylabel('Number of Trades')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('logs/entry_time_distribution.png')
        
        self.logger.info("Performance plots generated")


def main():
    """Main function for running the backtesting script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest Ross Cameron trading strategy')
    
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital for backtest')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create backtest engine
    engine = BacktestEngine(
        config_file=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital
    )
    
    # Run backtest
    results = engine.run_backtest()
    
    # Print summary results
    print("\n=== Backtest Results ===")
    print(f"Period: {results['summary']['start_date']} to {results['summary']['end_date']}")
    print(f"Initial Capital: ${results['summary']['initial_capital']:.2f}")
    print(f"Final Capital: ${results['summary']['final_capital']:.2f}")
    print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['summary']['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['summary']['total_trades']}")
    print(f"Win Rate: {results['summary']['win_rate']:.2f}")
    print(f"Profit Factor: {results['summary']['profit_factor']:.2f}")
    
    # Print pattern-specific statistics if available
    if results['trade_history']:
        print("\n=== Pattern Performance ===")
        pattern_stats = {}
        
        for trade in results['trade_history']:
            pattern = trade.get('pattern', 'unknown')
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0.0,
                    'loss': 0.0
                }
            
            pattern_stats[pattern]['count'] += 1
            realized_pnl = trade.get('realized_pnl', 0.0)
            
            if realized_pnl > 0:
                pattern_stats[pattern]['wins'] += 1
                pattern_stats[pattern]['profit'] += realized_pnl
            else:
                pattern_stats[pattern]['losses'] += 1
                pattern_stats[pattern]['loss'] += realized_pnl
        
        for pattern, stats in pattern_stats.items():
            win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            net_pnl = stats['profit'] + stats['loss']
            profit_factor = abs(stats['profit'] / stats['loss']) if stats['loss'] < 0 else float('inf')
            
            print(f"\n{pattern.title()} Pattern:")
            print(f"  Trades: {stats['count']}")
            print(f"  Win Rate: {win_rate:.2f}")
            print(f"  Net P&L: ${net_pnl:.2f}")
            print(f"  Profit Factor: {profit_factor:.2f}")


if __name__ == '__main__':
    main()