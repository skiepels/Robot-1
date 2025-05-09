"""
Backtesting Engine

This script runs backtesting for the trading strategy on historical data
retrieved from Interactive Brokers.
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
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.ib_connector import IBConnector
from src.conditions.condition1_price import PriceCondition
from src.conditions.condition2_percent_up import PercentUpCondition
from src.conditions.condition3_volume import VolumeCondition
from src.conditions.condition4_news import NewsCondition
from src.conditions.condition5_float import FloatCondition
from src.entry.candlestick import CandlestickPatterns
from src.trading.risk_manager import RiskManager
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Custom JSON Encoder for handling datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)


class BacktestEngine:
    """Backtesting engine for the trading strategy."""
    
    def __init__(self, config_file='config/config.yaml', start_date=None, end_date=None, 
                initial_capital=10000.0, symbols=None):
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
        symbols: list
            List of symbols to backtest
        """
        self.config_file = config_file
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=5)
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        self.initial_capital = initial_capital or float(os.getenv('INITIAL_CAPITAL', 10000.0))
        self.symbols = symbols or ['GPUS']  # Default to GPUS if no symbols provided
        
        # Setup logger
        self.logger = setup_logger('backtest', log_dir='logs', console_level=logging.INFO)
        
        # Initialize components
        self.ib_connector = None
        self.price_condition = None
        self.percent_up_condition = None
        self.volume_condition = None
        self.news_condition = None
        self.float_condition = None
        self.candlestick_patterns = None
        self.risk_manager = None
        
        # Results storage
        self.daily_results = {}
        self.trade_history = []
        self.equity_curve = []
        
        self.logger.info(f"Backtest initialized from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def setup_components(self):
        """Initialize strategy components."""
        self.logger.info("Setting up strategy components")
        
        # Initialize IB connector
        self.ib_connector = IBConnector()
        if not self.ib_connector.connect():
            self.logger.error("Failed to connect to Interactive Brokers")
            return False
        
        # Initialize trading conditions
        self.price_condition = PriceCondition(min_price=2.0, max_price=20.0)
        self.percent_up_condition = PercentUpCondition(min_percent=10.0)
        self.volume_condition = VolumeCondition(min_rel_volume=5.0)
        self.news_condition = NewsCondition()
        self.float_condition = FloatCondition(max_float=20_000_000)
        
        # Initialize candlestick pattern detector
        self.candlestick_patterns = CandlestickPatterns()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_risk_per_trade_pct=float(os.getenv('MAX_RISK_PER_TRADE_PCT', 1.0)),
            daily_max_loss_pct=float(os.getenv('DAILY_MAX_LOSS_PCT', 3.0)),
            profit_loss_ratio=float(os.getenv('PROFIT_LOSS_RATIO', 2.0)),
            max_open_positions=int(os.getenv('MAX_OPEN_POSITIONS', 3))
        )
        
        self.logger.info("Strategy components initialized")
        return True
    
    def run_backtest(self):
        """Run the backtest through the specified date range."""
        self.logger.info("Starting backtest")
        
        # Setup components
        if not self.setup_components():
            self.logger.error("Failed to setup components, aborting backtest")
            return None
        
        # Process each symbol
        for symbol in self.symbols:
            self.logger.info(f"Processing symbol: {symbol}")
            
            # Get historical data
            data = self.get_historical_data(symbol)
            if data.empty:
                self.logger.warning(f"No historical data found for {symbol}, skipping")
                continue
            
            # Process data day by day
            self.process_symbol_data(symbol, data)
        
        # Generate and return results
        results = self.generate_results()
        
        # Disconnect from IB
        if self.ib_connector:
            self.ib_connector.disconnect()
        
        return results
    
    def get_historical_data(self, symbol):
        """
        Get historical data for a symbol from Interactive Brokers.
        
        Parameters:
        -----------
        symbol: str
            Symbol to get data for
            
        Returns:
        --------
        pandas.DataFrame: Historical data
        """
        # Calculate duration based on start and end dates
        days_diff = (self.end_date - self.start_date).days + 1
        duration = f"{days_diff} D"
        
        self.logger.info(f"Requesting {duration} of historical data for {symbol}")
        
        # Get 1-minute data for detailed analysis
        df = self.ib_connector.get_historical_data(
            symbol, 
            duration=duration, 
            bar_size='1 min', 
            what_to_show='TRADES', 
            use_rth=True
        )
        
        if df.empty:
            self.logger.warning(f"No data returned for {symbol}")
            return df
        
        self.logger.info(f"Retrieved {len(df)} data points for {symbol}")
        
        # Add daily change percentage
        if 'close' in df.columns:
            # Group by date to get daily open/close
            daily = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate daily percent change
            daily['prev_close'] = daily['close'].shift(1)
            daily['day_change_pct'] = (daily['close'] / daily['prev_close'] - 1) * 100
            
            # Merge back to minute data
            dates = df.index.floor('D')
            df['date_column'] = dates
            df = pd.merge(df, daily[['day_change_pct']], left_on='date_column', right_index=True, how='left')
            df.drop('date_column', axis=1, inplace=True)
        
        return df
    
    def process_symbol_data(self, symbol, data):
        """
        Process historical data for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Symbol to process
        data: pandas.DataFrame
            Historical data for the symbol
        """
        # Group data by day
        dates = data.index.floor('D').unique()
        
        # Process each day
        for date in dates:
            # Get data for this day
            day_data = data[data.index.floor('D') == date]
            if day_data.empty:
                continue
            
            self.logger.info(f"Processing {symbol} for {date.strftime('%Y-%m-%d')}")
            
            # Reset daily tracking
            self.risk_manager.reset_daily_metrics()
            daily_stats = {
                'trades_taken': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'gross_pnl': 0.0
            }
            
            # Check if the symbol meets our conditions on this day
            day_open = day_data['open'].iloc[0]
            day_high = day_data['high'].max()
            day_low = day_data['low'].min()
            day_close = day_data['close'].iloc[-1]
            day_volume = day_data['volume'].sum()
            day_change_pct = day_data['day_change_pct'].iloc[0]
            
            # Check conditions
            price_ok = self.price_condition.check(day_close)
            percent_up_ok = self.percent_up_condition.check(day_change_pct)
            volume_ok = True  # We don't have volume comparison data in backtest
            float_ok = self.float_condition.check(20_000_000)  # Placeholder, we don't have float data
            news_ok = True  # We don't have news data in backtest
            
            conditions_met = price_ok and percent_up_ok and volume_ok and float_ok and news_ok
            
            if not conditions_met:
                self.logger.info(f"{symbol} does not meet conditions on {date.strftime('%Y-%m-%d')}")
                continue
            
            self.logger.info(f"{symbol} meets conditions on {date.strftime('%Y-%m-%d')}")
            
            # Simulate trading day
            trades_results = self.simulate_trading_day(symbol, day_data)
            
            # Update daily stats
            daily_stats['trades_taken'] = len(trades_results)
            daily_stats['winning_trades'] = sum(1 for t in trades_results if t['pnl'] > 0)
            daily_stats['losing_trades'] = sum(1 for t in trades_results if t['pnl'] <= 0)
            daily_stats['total_profit'] = sum(t['pnl'] for t in trades_results if t['pnl'] > 0)
            daily_stats['total_loss'] = sum(t['pnl'] for t in trades_results if t['pnl'] <= 0)
            daily_stats['gross_pnl'] = sum(t['pnl'] for t in trades_results)
            
            # Store daily results
            day_str = date.strftime('%Y-%m-%d')
            self.daily_results[day_str] = {
                'symbol': symbol,
                'trades_taken': daily_stats['trades_taken'],
                'winning_trades': daily_stats['winning_trades'],
                'losing_trades': daily_stats['losing_trades'],
                'win_rate': (daily_stats['winning_trades'] / daily_stats['trades_taken'] 
                          if daily_stats['trades_taken'] > 0 else 0),
                'total_profit': daily_stats['total_profit'],
                'total_loss': daily_stats['total_loss'],
                'net_pnl': daily_stats['gross_pnl'],
                'equity': self.risk_manager.current_capital,
                'open': day_open,
                'high': day_high,
                'low': day_low,
                'close': day_close,
                'volume': day_volume,
                'change_pct': day_change_pct
            }
            
            # Update equity curve
            self.equity_curve.append({
                'date': day_str,
                'equity': self.risk_manager.current_capital
            })
            
            # Store trade history
            self.trade_history.extend(trades_results)
    
    def simulate_trading_day(self, symbol, day_data):
        """
        Simulate trading for a single day using triple candlestick patterns.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        day_data: pandas.DataFrame
            Data for the day
                
        Returns:
        --------
        list: List of trade results
        """
        trades = []
        active_trade = None
        macd_crossover_confirmed = False
        
        # Calculate MACD for the day's data
        day_data['ema12'] = day_data['close'].ewm(span=12, adjust=False).mean()
        day_data['ema26'] = day_data['close'].ewm(span=26, adjust=False).mean()
        day_data['macd_line'] = day_data['ema12'] - day_data['ema26']
        day_data['macd_signal'] = day_data['macd_line'].ewm(span=9, adjust=False).mean()
        day_data['macd_histogram'] = day_data['macd_line'] - day_data['macd_signal']
        
        # Loop through each minute of the day
        for i in range(5, len(day_data)):  # Start from 5 to have enough data for 5-candle patterns
            # Get current and previous minute data
            current = day_data.iloc[i]
            previous_minutes = day_data.iloc[max(0, i-10):i+1]
            
            # Check for MACD crossover confirmation
            # MACD line crosses above signal line and stays above for at least 3 periods
            if (not macd_crossover_confirmed and 
                i >= 3 and 
                all(day_data.iloc[i-j]['macd_line'] > day_data.iloc[i-j]['macd_signal'] for j in range(3))):
                macd_crossover_confirmed = True
                self.logger.info(f"MACD crossover confirmed at {current.name}")
                
                # If we have an active trade during MACD crossover, close it
                if active_trade is not None:
                    # Calculate PnL
                    exit_price = current['close']
                    shares = active_trade['shares']
                    pnl = (exit_price - active_trade['entry_price']) * shares
                    
                    # Record trade result
                    trade_result = {
                        'symbol': symbol,
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current.name,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'exit_reason': 'macd_crossover',
                        'pattern': active_trade['pattern'],
                        'duration': (current.name - active_trade['entry_time']).total_seconds() / 60
                    }
                    
                    trades.append(trade_result)
                    active_trade = None
                    
                    # Update risk manager
                    self.risk_manager.update_account_metrics(pnl, is_win=pnl > 0)
                    
                    self.logger.info(f"Closed trade at MACD crossover: {symbol} at ${exit_price:.2f}, PnL: ${pnl:.2f}")
            
            # If MACD crossover confirmed, check if volume has returned
            # (volume spike above 20-day average) to reset the flag
            if macd_crossover_confirmed and 'volume' in day_data.columns:
                # Calculate 20-period volume average
                vol_avg = day_data.iloc[max(0, i-20):i]['volume'].mean()
                
                # Check for volume spike (150% of average)
                if current['volume'] > vol_avg * 1.5:
                    macd_crossover_confirmed = False
                    self.logger.info(f"Volume returned at {current.name}, resuming trading")
            
            # If no active trade and MACD crossover not confirmed, check for entry signals
            if active_trade is None and not macd_crossover_confirmed:
                # Get a window of candles for pattern detection
                pattern_window = day_data.iloc[max(0, i-10):i+1]
                
                # Check for triple candlestick patterns
                entry_signals = self.candlestick_patterns.detect_entry_signal(pattern_window)
                
                # If we have any entry signals
                if entry_signals:
                    # Choose the first pattern detected (prioritize bullish patterns)
                    pattern = None
                    
                    # Check bullish patterns first
                    for p in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three']:
                        if p in entry_signals:
                            pattern = p
                            break
                    
                    # If no bullish pattern found, check bearish patterns
                    if pattern is None:
                        for p in ['evening_star', 'evening_doji_star', 'three_black_crows', 'falling_three']:
                            if p in entry_signals:
                                pattern = p
                                break
                    
                    if pattern:
                        # Get entry and stop prices based on the pattern
                        entry_price = self.candlestick_patterns.get_optimal_entry_price(pattern_window, pattern)
                        stop_price = self.candlestick_patterns.get_optimal_stop_price(pattern_window, pattern)
                        
                        # Calculate target price (2:1 reward-to-risk ratio)
                        risk = abs(entry_price - stop_price)
                        target_price = entry_price + (risk * 2) if pattern in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] else entry_price - (risk * 2)
                        
                        # Calculate position size
                        shares = self.risk_manager.calculate_position_size(entry_price, stop_price)
                        
                        if shares > 0:
                            # Validate trade
                            is_valid, reason = self.risk_manager.validate_trade(
                                symbol, entry_price, stop_price, target_price, shares
                            )
                            
                            if is_valid:
                                # Create active trade
                                active_trade = {
                                    'symbol': symbol,
                                    'entry_time': current.name,
                                    'entry_price': entry_price,
                                    'stop_price': stop_price,
                                    'target_price': target_price,
                                    'shares': shares,
                                    'pattern': pattern
                                }
                                
                                self.logger.info(
                                    f"Entry signal ({pattern}) at {current.name}: {symbol} at ${entry_price:.2f}, "
                                    f"stop: ${stop_price:.2f}, target: ${target_price:.2f}, shares: {shares}"
                                )
            
            # If we have an active trade, check for exit signals
            elif active_trade is not None:
                # Check for stop loss hit
                if (active_trade['pattern'] in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] and current['low'] <= active_trade['stop_price']) or \
                   (active_trade['pattern'] in ['evening_star', 'evening_doji_star', 'three_black_crows', 'falling_three'] and current['high'] >= active_trade['stop_price']):
                    # Calculate PnL
                    exit_price = active_trade['stop_price']
                    shares = active_trade['shares']
                    pnl = (exit_price - active_trade['entry_price']) * shares if active_trade['pattern'] in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] else (active_trade['entry_price'] - exit_price) * shares
                    
                    # Record trade result
                    trade_result = {
                        'symbol': symbol,
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current.name,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'pattern': active_trade['pattern'],
                        'duration': (current.name - active_trade['entry_time']).total_seconds() / 60
                    }
                    
                    trades.append(trade_result)
                    active_trade = None
                    
                    # Update risk manager
                    self.risk_manager.update_account_metrics(pnl, is_win=False)
                    
                    self.logger.info(
                        f"Stop loss at {current.name}: {symbol} at ${exit_price:.2f}, "
                        f"PnL: ${pnl:.2f}"
                    )
                
                # Check for target hit
                elif (active_trade['pattern'] in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] and current['high'] >= active_trade['target_price']) or \
                     (active_trade['pattern'] in ['evening_star', 'evening_doji_star', 'three_black_crows', 'falling_three'] and current['low'] <= active_trade['target_price']):
                    # Calculate PnL
                    exit_price = active_trade['target_price']
                    shares = active_trade['shares']
                    pnl = (exit_price - active_trade['entry_price']) * shares if active_trade['pattern'] in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] else (active_trade['entry_price'] - exit_price) * shares
                    
                    # Record trade result
                    trade_result = {
                        'symbol': symbol,
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current.name,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'exit_reason': 'target_reached',
                        'pattern': active_trade['pattern'],
                        'duration': (current.name - active_trade['entry_time']).total_seconds() / 60
                    }
                    
                    trades.append(trade_result)
                    active_trade = None
                    
                    # Update risk manager
                    self.risk_manager.update_account_metrics(pnl, is_win=True)
                    
                    self.logger.info(
                        f"Target reached at {current.name}: {symbol} at ${exit_price:.2f}, "
                        f"PnL: ${pnl:.2f}"
                    )
                
                # Check for end of day exit (if we're in the last 15 minutes of the trading day)
                elif i >= len(day_data) - 15:
                    # Calculate PnL
                    exit_price = current['close']
                    shares = active_trade['shares']
                    pnl = (exit_price - active_trade['entry_price']) * shares if active_trade['pattern'] in ['morning_star', 'morning_doji_star', 'three_white_soldiers', 'rising_three'] else (active_trade['entry_price'] - exit_price) * shares
                    
                    # Record trade result
                    trade_result = {
                        'symbol': symbol,
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current.name,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'exit_reason': 'end_of_day',
                        'pattern': active_trade['pattern'],
                        'duration': (current.name - active_trade['entry_time']).total_seconds() / 60
                    }
                    
                    trades.append(trade_result)
                    active_trade = None
                    
                    # Update risk manager
                    self.risk_manager.update_account_metrics(pnl, is_win=pnl > 0)
                    
                    self.logger.info(
                        f"End of day exit at {current.name}: {symbol} at ${exit_price:.2f}, "
                        f"PnL: ${pnl:.2f}"
                    )
        
        return trades
    
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
        final_equity = self.risk_manager.current_capital if self.equity_curve else initial_equity
        
        total_return = ((final_equity / initial_equity) - 1) * 100
        
        # Calculate daily returns
        daily_returns = []
        if self.equity_curve:
            prev_equity = initial_equity
            for point in self.equity_curve:
                equity = point['equity']
                daily_return = (equity / prev_equity) - 1
                daily_returns.append(daily_return)
                prev_equity = equity
        
        # Calculate additional metrics
        trading_days = len(self.daily_results)
        if trading_days > 0 and total_return != 0:
            annualized_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100
            
            # Calculate Sharpe ratio
            if daily_returns:
                avg_daily_return = sum(daily_returns) / len(daily_returns)
                std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 0
                
                if std_daily_return > 0:
                    sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate drawdown
            max_drawdown = 0
            peak = initial_equity
            
            for point in self.equity_curve:
                equity = point['equity']
                if equity > peak:
                    peak = equity
                else:
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
        results_file = f"logs/backtest_results_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            # Use custom DateTimeEncoder to handle datetime objects
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Generate plots if there are trades
        if total_trades > 0:
            self._generate_performance_plots()
        
        # Print pattern-specific statistics if available
        if self.trade_history:
            print("\n=== Pattern Performance ===")
            pattern_stats = {}
            pattern_names = {
                'morning_star': 'Morning Star',
                'morning_doji_star': 'Morning Doji Star',
                'three_white_soldiers': 'Three White Soldiers',
                'rising_three': 'Rising Three',
                'evening_star': 'Evening Star',
                'evening_doji_star': 'Evening Doji Star',
                'three_black_crows': 'Three Black Crows',
                'falling_three': 'Falling Three'
            }
            
            for pattern in pattern_names.keys():
                pattern_trades = [trade for trade in self.trade_history if trade.get('pattern') == pattern]
                if pattern_trades:
                    count = len(pattern_trades)
                    wins = sum(1 for trade in pattern_trades if trade.get('pnl', 0) > 0)
                    losses = count - wins
                    profit = sum(trade.get('pnl', 0) for trade in pattern_trades if trade.get('pnl', 0) > 0)
                    loss = sum(trade.get('pnl', 0) for trade in pattern_trades if trade.get('pnl', 0) <= 0)
                    net_pnl = profit + loss
                    
                    pattern_stats[pattern_names[pattern]] = {
                        'count': count,
                        'wins': wins,
                        'losses': losses,
                        'win_rate': wins / count if count > 0 else 0,
                        'profit': profit,
                        'loss': loss,
                        'net_pnl': net_pnl,
                        'profit_factor': abs(profit / loss) if loss < 0 else float('inf'),
                        'avg_win': profit / wins if wins > 0 else 0,
                        'avg_loss': loss / losses if losses > 0 else 0,
                        'exit_reasons': {
                            'target_reached': sum(1 for trade in pattern_trades if trade.get('exit_reason') == 'target_reached'),
                            'stop_loss': sum(1 for trade in pattern_trades if trade.get('exit_reason') == 'stop_loss'),
                            'end_of_day': sum(1 for trade in pattern_trades if trade.get('exit_reason') == 'end_of_day'),
                            'macd_crossover': sum(1 for trade in pattern_trades if trade.get('exit_reason') == 'macd_crossover')
                        }
                    }
            
            # Add pattern statistics to results
            results['pattern_stats'] = pattern_stats
            
            # Print pattern-specific statistics
            for pattern_name, stats in pattern_stats.items():
                print(f"\n{pattern_name} Pattern:")
                print(f"  Trades: {stats['count']}")
                print(f"  Win Rate: {stats['win_rate']:.2f}")
                print(f"  Net P&L: ${stats['net_pnl']:.2f}")
                print(f"  Profit Factor: {stats['profit_factor']:.2f}")
                print(f"  Avg Win: ${stats['avg_win']:.2f}")
                print(f"  Avg Loss: ${stats['avg_loss']:.2f}")
                print(f"  Exit Reasons: Target {stats['exit_reasons']['target_reached']}, Stop {stats['exit_reasons']['stop_loss']}, EOD {stats['exit_reasons']['end_of_day']}, MACD {stats['exit_reasons']['macd_crossover']}")
        
        return results
    
    def _generate_performance_plots(self):
        """Generate performance plots from the backtest results."""
        # Create output directory
        os.makedirs('logs/plots', exist_ok=True)
        
        # Create equity curve plot
        if self.equity_curve:
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
            plt.savefig('logs/plots/equity_curve.png')
            plt.close()
        
        # Create daily P&L plot
        if self.daily_results:
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
            plt.savefig('logs/plots/daily_pnl.png')
            plt.close()
        
        # Create win rate plot
        if self.daily_results:
            plt.figure(figsize=(10, 6))
            dates = list(self.daily_results.keys())
            win_rates = [day['win_rate'] * 100 for day in self.daily_results.values()]
            trades_taken = [day['trades_taken'] for day in self.daily_results.values()]
            
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
            plt.savefig('logs/plots/win_rate.png')
            plt.close()
        
        # Create pattern distribution plot if we have trades
        if self.trade_history:
            patterns = [trade['pattern'] for trade in self.trade_history if 'pattern' in trade]
            if patterns:
                pattern_counts = {}
                for pattern in patterns:
                    # Convert pattern code to readable name
                    pattern_name = {
                        'morning_star': 'Morning Star',
                        'morning_doji_star': 'Morning Doji Star',
                        'three_white_soldiers': 'Three White Soldiers',
                        'rising_three': 'Rising Three',
                        'evening_star': 'Evening Star',
                        'evening_doji_star': 'Evening Doji Star',
                        'three_black_crows': 'Three Black Crows',
                        'falling_three': 'Falling Three'
                    }.get(pattern, pattern)
                    
                    pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                
                plt.figure(figsize=(10, 6))
                labels = list(pattern_counts.keys())
                sizes = list(pattern_counts.values())
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                      shadow=True, startangle=90)
                plt.axis('equal')
                plt.title('Trade Pattern Distribution')
                plt.tight_layout()
                plt.savefig('logs/plots/pattern_distribution.png')
                plt.close()
                
                # Create win rate by pattern plot
                pattern_win_rates = {}
                for pattern in set(patterns):
                    pattern_name = {
                        'morning_star': 'Morning Star',
                        'morning_doji_star': 'Morning Doji Star',
                        'three_white_soldiers': 'Three White Soldiers',
                        'rising_three': 'Rising Three',
                        'evening_star': 'Evening Star',
                        'evening_doji_star': 'Evening Doji Star',
                        'three_black_crows': 'Three Black Crows',
                        'falling_three': 'Falling Three'
                    }.get(pattern, pattern)
                    
                    pattern_trades = [t for t in self.trade_history if t.get('pattern') == pattern]
                    wins = sum(1 for t in pattern_trades if t.get('pnl', 0) > 0)
                    total = len(pattern_trades)
                    win_rate = wins / total if total > 0 else 0
                    pattern_win_rates[pattern_name] = win_rate * 100
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(list(pattern_win_rates.keys()), list(pattern_win_rates.values()), color='skyblue')
                plt.axhline(y=50, color='r', linestyle='-', alpha=0.3)  # 50% reference line
                plt.title('Win Rate by Pattern')
                plt.xlabel('Pattern')
                plt.ylabel('Win Rate (%)')
                plt.xticks(rotation=45)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('logs/plots/pattern_win_rates.png')
                plt.close()
        
        self.logger.info("Performance plots generated in logs/plots/ directory")