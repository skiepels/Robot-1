# src/trading/trading_bot.py
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

from src.data.ib_connector import IBConnector
from src.entry.candlestick import CandlestickPatterns
from src.trading.risk_manager import RiskManager

# Configuration
class Config:
    # Stock selection parameters
    MIN_PRICE = 2.0
    MAX_PRICE = 20.0
    MIN_PERCENT_UP = 10.0
    MIN_REL_VOLUME = 5.0
    MAX_FLOAT = 10_000_000
    
    # Risk management
    MAX_RISK_PER_TRADE_PCT = 1.0
    DAILY_MAX_LOSS_PCT = 3.0
    PROFIT_LOSS_RATIO = 2.0
    MAX_OPEN_POSITIONS = 3
    
    # EMA settings
    EMA_FAST = 9
    EMA_MID = 20
    EMA_SLOW = 200
    
    # MACD settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9


class TradingBot:
    def __init__(self, initial_capital=10000.0):
        """Initialize the trading bot."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        
        # Initialize components
        self.candlestick_patterns = CandlestickPatterns()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=Config.MAX_RISK_PER_TRADE_PCT,
            daily_max_loss_pct=Config.DAILY_MAX_LOSS_PCT,
            profit_loss_ratio=Config.PROFIT_LOSS_RATIO,
            max_open_positions=Config.MAX_OPEN_POSITIONS
        )
        
        # Set up logging
        self.setup_logger()
        
        self.logger.info(f"Trading Bot initialized with ${initial_capital:.2f}")
    
    def setup_logger(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)
        
        # Create file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for a DataFrame."""
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate EMAs
        result['ema9'] = result['close'].ewm(span=Config.EMA_FAST, adjust=False).mean()
        result['ema20'] = result['close'].ewm(span=Config.EMA_MID, adjust=False).mean()
        result['ema200'] = result['close'].ewm(span=Config.EMA_SLOW, adjust=False).mean()
        
        # Calculate MACD
        result['macd_fast'] = result['close'].ewm(span=Config.MACD_FAST, adjust=False).mean()
        result['macd_slow'] = result['close'].ewm(span=Config.MACD_SLOW, adjust=False).mean()
        result['macd_line'] = result['macd_fast'] - result['macd_slow']
        result['macd_signal'] = result['macd_line'].ewm(span=Config.MACD_SIGNAL, adjust=False).mean()
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

    def detect_first_pullback(self, df):
        """Detect the first pullback pattern."""
        # Need at least 7 candles to identify pattern
        if len(df) < 7:
            return None
            
        # Get recent data
        recent_data = df.iloc[-7:]
        
        # Check for initial surge (at least 2 strong green candles)
        initial_candles = recent_data.iloc[0:3]
        green_candles = sum(1 for i in range(len(initial_candles)) if initial_candles.iloc[i]['close'] > initial_candles.iloc[i]['open'])
        
        if green_candles < 2:
            return None  # No initial surge
            
        # Check for pullback (at least 1 red candle)
        pullback_candles = recent_data.iloc[3:6]
        red_candles = sum(1 for i in range(len(pullback_candles)) if pullback_candles.iloc[i]['close'] < pullback_candles.iloc[i]['open'])
        
        if red_candles < 1:
            return None  # No pullback
        
        # Check the last candle for potential entry (candle over candle)
        last_candle = recent_data.iloc[-1]
        prev_candle = recent_data.iloc[-2]
        
        is_candle_over_candle = last_candle['high'] > prev_candle['high'] and last_candle['close'] > last_candle['open']
        
        # Verify with MACD
        if 'macd_line' not in last_candle:
            return None
        
        macd_positive = last_candle['macd_line'] > 0
        
        # Check volume (should be increasing on the breakout)
        volume_increasing = last_candle['volume'] > prev_candle['volume'] if 'volume' in last_candle else True
        
        if is_candle_over_candle and macd_positive and volume_increasing:
            # Calculate support (stop loss level)
            pullback_low = min(pullback_candles['low'])
            
            return {
                'pattern': 'first_pullback',
                'entry_price': last_candle['high'],  # Entry above the current candle's high
                'stop_price': pullback_low,
                'is_valid': True
            }
            
        return None

    def detect_micro_pullback(self, df):
        """Detect the micro pullback pattern."""
        # Need at least 5 candles to identify pattern
        if len(df) < 5:
            return None
            
        # Get recent data
        recent_data = df.iloc[-5:]
        
        # Check for strong uptrend (first 3 candles)
        first_3_candles = recent_data.iloc[:3]
        uptrend = first_3_candles['close'].iloc[-1] > first_3_candles['close'].iloc[0]
        
        if not uptrend:
            return None
            
        # Look for a micro pullback (1-2 candles)
        pullback_candle = recent_data.iloc[-2]
        is_pullback = pullback_candle['close'] < pullback_candle['open']
        
        # Check for the bottoming tail
        body_size = abs(pullback_candle['close'] - pullback_candle['open'])
        lower_wick = min(pullback_candle['open'], pullback_candle['close']) - pullback_candle['low']
        has_bottoming_tail = lower_wick > body_size * 0.5
        
        # Check for entry candle (candle over candle)
        last_candle = recent_data.iloc[-1]
        is_candle_over_candle = last_candle['high'] > pullback_candle['high'] and last_candle['close'] > last_candle['open']
        
        # Verify with MACD
        if 'macd_line' not in last_candle:
            return None
            
        macd_positive = last_candle['macd_line'] > 0
        
        # Check if pullback stayed above 9 EMA
        if 'ema9' not in pullback_candle:
            return None
            
        above_ema = pullback_candle['low'] > pullback_candle['ema9']
        
        if (is_pullback or has_bottoming_tail) and is_candle_over_candle and macd_positive and above_ema:
            return {
                'pattern': 'micro_pullback',
                'entry_price': last_candle['high'],  # Entry above the current candle's high
                'stop_price': pullback_candle['low'],
                'is_valid': True
            }
            
        return None

    def detect_new_high_breakout(self, df):
        """Detect the new high breakout pattern."""
        # Need at least 6 candles to identify pattern
        if len(df) < 6:
            return None
            
        # Get recent data
        recent_data = df.iloc[-6:]
        
        # Find the previous high (excluding most recent candle)
        previous_high = recent_data.iloc[:-1]['high'].max()
        
        # Check the last candle for breakout
        last_candle = recent_data.iloc[-1]
        is_breakout = last_candle['high'] > previous_high and last_candle['close'] > last_candle['open']
        
        # Verify with MACD
        if 'macd_line' not in last_candle:
            return None
            
        macd_positive = last_candle['macd_line'] > 0
        
        # Check volume (should be increasing on the breakout)
        prev_candle = recent_data.iloc[-2]
        volume_increasing = last_candle['volume'] > prev_candle['volume'] if 'volume' in last_candle else True
        
        # Check for prior consolidation (tight range)
        prior_candles = recent_data.iloc[1:-1]
        high_range = prior_candles['high'].max() - prior_candles['high'].min()
        price_range_percent = high_range / prior_candles['close'].mean() * 100
        
        is_consolidated = price_range_percent < 3.0  # Less than 3% range is considered consolidation
        
        if is_breakout and macd_positive and volume_increasing and is_consolidated:
            return {
                'pattern': 'new_high_breakout',
                'entry_price': last_candle['high'],  # Entry above the current candle's high
                'stop_price': min(recent_data.iloc[-3:]['low']),  # Stop below recent lows
                'is_valid': True
            }
            
        return None
    
    def backtest_strategy(self, ib_connector, symbol='GPUS', start_date=None, end_date=None):
        """Run a backtest using Ross Cameron's strategy."""
        self.logger.info(f"Starting backtest for {symbol}")
        
        # Format dates
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        self.logger.info(f"Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Calculate duration
        days_diff = (end_date - start_date).days + 1
        duration = f"{days_diff} D"
        
        self.logger.info(f"Requesting {duration} of historical data for {symbol}")
        
        # Get historical data from IB
        df = ib_connector.get_historical_data(
            symbol, 
            duration=duration, 
            bar_size='1 min', 
            what_to_show='TRADES', 
            use_rth=True
        )
        
        if df.empty:
            self.logger.warning(f"No historical data returned for {symbol}")
            return None
            
        self.logger.info(f"Retrieved {len(df)} data points for {symbol}")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Reset performance metrics
        self.risk_manager.reset_daily_metrics()
        self.trade_history = []
        
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
            
            # Check if the stock meets all 5 conditions
            day_open = day_data['open'].iloc[0]
            day_high = day_data['high'].max()
            day_low = day_data['low'].min()
            day_close = day_data['close'].iloc[-1]
            day_volume = day_data['volume'].sum()
            
            # Calculate daily percent change
            # Note: In a real implementation, you would need to fetch the previous day's close
            # Here we'll simulate it by using the open price and assuming a gap
            prev_close = day_open * 0.9  # Simulate a 10% gap up
            day_change_pct = (day_close / prev_close - 1) * 100
            
            # Check conditions
            price_ok = Config.MIN_PRICE <= day_close <= Config.MAX_PRICE
            percent_up_ok = day_change_pct >= Config.MIN_PERCENT_UP
            volume_ok = True  # Assume volume condition is met for this backtest
            float_ok = True   # Assume float condition is met for this backtest
            news_ok = True    # Assume news condition is met for this backtest
            
            all_conditions_met = price_ok and percent_up_ok and volume_ok and float_ok and news_ok
            
            if not all_conditions_met:
                self.logger.info(f"{symbol} does not meet all 5 conditions on {date.strftime('%Y-%m-%d')}")
                continue
                
            self.logger.info(f"{symbol} meets all 5 conditions on {date.strftime('%Y-%m-%d')}")
            
            # Simulate trading for this day
            self._simulate_trading_day(symbol, day_data)
        
        # Generate results
        results = self._generate_backtest_results()
        
        return results

    def _simulate_trading_day(self, symbol, day_data):
        """Simulate trading for a single day."""
        trading_started = False
        stop_trading = False
        
        # Loop through each minute of the day
        for i in range(3, len(day_data)):
            if stop_trading:
                break
                
            # Analyze a window of recent data
            window = day_data.iloc[max(0, i-10):i+1]
            
            # If we haven't started trading yet, check if conditions are met
            if not trading_started:
                # Look for entry patterns
                patterns = []
                
                # Check for first pullback pattern
                first_pullback = self.detect_first_pullback(window)
                if first_pullback and first_pullback['is_valid']:
                    patterns.append(first_pullback)
                    
                # Check for micro pullback pattern
                micro_pullback = self.detect_micro_pullback(window)
                if micro_pullback and micro_pullback['is_valid']:
                    patterns.append(micro_pullback)
                    
                # Check for new high breakout pattern
                new_high_breakout = self.detect_new_high_breakout(window)
                if new_high_breakout and new_high_breakout['is_valid']:
                    patterns.append(new_high_breakout)
                
                # If we found a valid pattern, start trading
                if patterns:
                    trading_started = True
                    
                    # Choose the best pattern (prioritize first pullback)
                    best_pattern = None
                    for pattern in patterns:
                        if pattern['pattern'] == 'first_pullback':
                            best_pattern = pattern
                            break
                    
                    if not best_pattern:
                        for pattern in patterns:
                            if pattern['pattern'] == 'new_high_breakout':
                                best_pattern = pattern
                                break
                    
                    if not best_pattern:
                        best_pattern = patterns[0]
                    
                    # Execute trade with this pattern
                    self._execute_simulated_trade(symbol, window, best_pattern)
            else:
                # We're already trading, look for additional setups
                
                # Check for market reversal
                current = window.iloc[-1]
                if 'macd_line' in current and 'ema9' in current:
                    # 1. MACD negative
                    macd_negative = current['macd_line'] < 0
                    
                    # 2. Price below 9 EMA
                    below_9ema = current['close'] < current['ema9']
                    
                    # 3. Red candle
                    is_red = current['close'] < current['open']
                    
                    # If all conditions are met, stop trading
                    if macd_negative and below_9ema and is_red:
                        self.logger.info(f"Market reversal detected, stopping trading for the day.")
                        stop_trading = True
                        continue
                
                # Look for new trading opportunities
                patterns = []
                
                # Check for first pullback pattern
                first_pullback = self.detect_first_pullback(window)
                if first_pullback and first_pullback['is_valid']:
                    patterns.append(first_pullback)
                    
                # Check for micro pullback pattern
                micro_pullback = self.detect_micro_pullback(window)
                if micro_pullback and micro_pullback['is_valid']:
                    patterns.append(micro_pullback)
                    
                # Check for new high breakout pattern
                new_high_breakout = self.detect_new_high_breakout(window)
                if new_high_breakout and new_high_breakout['is_valid']:
                    patterns.append(new_high_breakout)
                
                # If we found a valid pattern, consider a new trade
                if patterns and len(self.active_trades) < Config.MAX_OPEN_POSITIONS:
                    # Choose the best pattern
                    best_pattern = None
                    for pattern in patterns:
                        if pattern['pattern'] == 'first_pullback':
                            best_pattern = pattern
                            break
                    
                    if not best_pattern:
                        for pattern in patterns:
                            if pattern['pattern'] == 'new_high_breakout':
                                best_pattern = pattern
                                break
                    
                    if not best_pattern:
                        best_pattern = patterns[0]
                    
                    # Execute trade with this pattern
                    self._execute_simulated_trade(symbol, window, best_pattern)
            
            # Manage active trades
            self._manage_simulated_trades(symbol, window.iloc[-1])

    def _execute_simulated_trade(self, symbol, window, pattern):
        """Execute a simulated trade based on the pattern."""
        # Get current data
        current_candle = window.iloc[-1]
        
        # Calculate position size
        entry_price = pattern['entry_price']
        stop_price = pattern['stop_price']
        
        # Ensure stop is at least 2% below entry
        min_stop_distance = entry_price * 0.02
        if entry_price - stop_price < min_stop_distance:
            stop_price = entry_price - min_stop_distance
        
        # Calculate position size based on risk
        risk_per_share = entry_price - stop_price
        max_dollar_risk = self.current_capital * (Config.MAX_RISK_PER_TRADE_PCT / 100)
        shares = int(max_dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        
        if shares <= 0:
            return
        
        # Calculate target price (2:1 reward-to-risk ratio)
        target_price = entry_price + (risk_per_share * Config.PROFIT_LOSS_RATIO)
        
        # Generate a unique trade ID
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.trade_history)}"
        
        # Create trade record
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'pattern': pattern['pattern'],
            'entry_time': current_candle.name,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'status': 'open'
        }
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        self.logger.info(f"Entry signal ({pattern['pattern']}) at {current_candle.name}: "
                       f"{symbol} at ${entry_price:.2f}, stop: ${stop_price:.2f}, "
                       f"target: ${target_price:.2f}, shares: {shares}")

    def _manage_simulated_trades(self, symbol, current_candle):
        """Manage active simulated trades."""
        trades_to_close = []
        
        for trade_id, trade in self.active_trades.items():
            if trade['symbol'] != symbol:
                continue
                
            # Check if stop loss hit
            if current_candle['low'] <= trade['stop_price']:
                # Calculate P&L
                exit_price = trade['stop_price']
                shares = trade['shares']
                pnl = (exit_price - trade['entry_price']) * shares
                
                # Record trade result
                trade_result = trade.copy()
                trade_result['exit_time'] = current_candle.name
                trade_result['exit_price'] = exit_price
                trade_result['pnl'] = pnl
                trade_result['exit_reason'] = 'stop_loss'
                trade_result['status'] = 'closed'
                
                # Add to trade history
                self.trade_history.append(trade_result)
                
                # Update account
                self.current_capital += pnl
                
                # Mark for removal
                trades_to_close.append(trade_id)
                
                self.logger.info(f"Stop loss at {current_candle.name}: {symbol} at ${exit_price:.2f}, "
                               f"PnL: ${pnl:.2f}")
                continue
            
            # Check if target reached
            if current_candle['high'] >= trade['target_price']:
                # Calculate P&L
                exit_price = trade['target_price']
                shares = trade['shares']
                pnl = (exit_price - trade['entry_price']) * shares
                
                # Record trade result
                trade_result = trade.copy()
                trade_result['exit_time'] = current_candle.name
                trade_result['exit_price'] = exit_price
                trade_result['pnl'] = pnl
                trade_result['exit_reason'] = 'target_reached'
                trade_result['status'] = 'closed'
                
                # Add to trade history
                self.trade_history.append(trade_result)
                
                # Update account
                self.current_capital += pnl
                
                # Mark for removal
                trades_to_close.append(trade_id)
                
                self.logger.info(f"Target reached at {current_candle.name}: {symbol} at ${exit_price:.2f}, "
                               f"PnL: ${pnl:.2f}")
                continue
        
        # Remove closed trades
        for trade_id in trades_to_close:
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]

    def _generate_backtest_results(self):
        """Generate and return backtest results."""
        if not self.trade_history:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_pnl': 0,
                'return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trade_history': []
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        total_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        total_loss = sum(t['pnl'] for t in self.trade_history if t['pnl'] <= 0)
        total_pnl = total_profit + total_loss
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate metrics by pattern
        pattern_stats = {}
        
        for trade in self.trade_history:
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
            'trade_history': self.trade_history,
            'pattern_stats': pattern_stats
        }
        
        return results

    def set_watch_list(self, symbols):
        """Set the watch list for live trading."""
        self.watch_list = symbols
        self.logger.info(f"Watch list set to: {', '.join(symbols)}")