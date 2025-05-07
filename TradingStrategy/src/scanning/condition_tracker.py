"""
Condition Tracker

This module monitors market conditions and tracks when stocks meet specific
trading criteria based on Ross Cameron's day trading approach.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
from src.entry.candlestick import CandlestickPatterns

logger = logging.getLogger(__name__)


class ConditionTracker:
    def __init__(self, market_data_provider, news_provider):
        """
        Initialize the condition tracker.
        
        Parameters:
        -----------
        market_data_provider: MarketDataProvider
            Provider for market data
        news_provider: NewsDataProvider
            Provider for news data
        """
        self.market_data = market_data_provider
        self.news = news_provider
        self.pattern_detector = CandlestickPatterns()
        self.tracked_stocks = {}  # Dictionary to track stocks that meet criteria
        self.alert_history = []  # History of alerts generated
        
        # Trading session info
        self.market_open = False
        self.pre_market = False
        self.post_market = False
        
        # Condition flags
        self.strong_market = False  # Flag for strong overall market conditions
        
        # Initialize tracking of specific patterns
        self.bull_flags = {}
        self.micro_pullbacks = {}
        self.new_high_breakouts = {}
    
    def update_market_session(self):
        """Update market session status (pre-market, regular, after-hours)"""
        now = datetime.now()
        
        # Regular market hours (9:30 AM - 4:00 PM Eastern)
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Pre-market hours (4:00 AM - 9:30 AM Eastern)
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        
        # After-hours (4:00 PM - 8:00 PM Eastern)
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        self.market_open = market_open_time <= now < market_close_time
        self.pre_market = pre_market_start <= now < market_open_time
        self.post_market = market_close_time <= now < after_hours_end
        
        logger.info(f"Updated market session: open={self.market_open}, pre={self.pre_market}, post={self.post_market}")
    
    def update_market_conditions(self):
        """
        Update overall market condition flags based on major indices and breadth.
        This helps determine the general trading environment.
        """
        try:
            # Get data for major indices (S&P 500, Nasdaq, Russell 2000)
            indices_data = {
                'SPY': self.market_data.get_intraday_data('SPY', interval='5m', lookback_days=1),
                'QQQ': self.market_data.get_intraday_data('QQQ', interval='5m', lookback_days=1),
                'IWM': self.market_data.get_intraday_data('IWM', interval='5m', lookback_days=1)
            }
            
            # Assess overall market strength
            strength_scores = []
            
            for symbol, data in indices_data.items():
                if data.empty:
                    continue
                
                # Calculate performance metrics for each index
                latest_data = data.iloc[-5:]  # Last 5 bars
                
                # Trend direction: up or down
                trend_up = latest_data['close'].iloc[-1] > latest_data['close'].iloc[0]
                
                # Above/below VWAP
                above_vwap = latest_data['close'].iloc[-1] > latest_data['vwap'].iloc[-1] if 'vwap' in latest_data.columns else False
                
                # Percentage change today
                pct_change = (latest_data['close'].iloc[-1] / latest_data['open'].iloc[0] - 1) * 100
                
                # Recent momentum
                momentum_up = latest_data['close'].iloc[-1] > latest_data['close'].iloc[-2]
                
                # Assign strength score
                score = 0
                score += 1 if trend_up else -1
                score += 1 if above_vwap else -1
                score += 1 if pct_change > 0.5 else (-1 if pct_change < -0.5 else 0)
                score += 1 if momentum_up else -1
                
                strength_scores.append(score)
            
            # Overall market strength based on average score
            avg_score = sum(strength_scores) / len(strength_scores) if strength_scores else 0
            
            # Strong market requires a positive average score
            self.strong_market = avg_score > 1
            
            logger.info(f"Updated market conditions: strong={self.strong_market}, score={avg_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
    
    def scan_for_trading_conditions(self, candidate_stocks, min_gap_pct=5.0, min_rel_volume=5.0, max_float=10_000_000):
        """
        Scan a list of candidate stocks for specific trading conditions.
        
        Parameters:
        -----------
        candidate_stocks: list
            List of Stock objects to scan
        min_gap_pct: float
            Minimum gap percentage to consider
        min_rel_volume: float
            Minimum relative volume to consider
        max_float: int
            Maximum shares in the float to consider
            
        Returns:
        --------
        dict: Dictionary of stocks that meet the trading conditions, with the condition as value
        """
        logger.info(f"Scanning {len(candidate_stocks)} stocks for trading conditions...")
        
        # Reset tracking for this scan
        self.tracked_stocks = {}
        self.bull_flags = {}
        self.micro_pullbacks = {}
        self.new_high_breakouts = {}
        
        # First, filter stocks based on Ross Cameron's key criteria
        filtered_stocks = []
        for stock in candidate_stocks:
            # Check if stock meets all basic criteria
            if not stock.meets_criteria(min_gap_pct=min_gap_pct, 
                                      min_rel_volume=min_rel_volume, 
                                      max_float=max_float):
                continue
            
            # Check for news catalyst
            news_items = self.news.get_stock_news(stock.symbol, days=1)
            if not news_items:
                continue
                
            stock.has_news = True
            stock.news_headline = news_items[0].headline
            stock.news_source = news_items[0].source
            stock.news_timestamp = news_items[0].date
            
            filtered_stocks.append(stock)
        
        logger.info(f"Found {len(filtered_stocks)} stocks that meet all basic criteria")
        
        # Now, check for specific trading patterns on the filtered stocks
        for stock in filtered_stocks:
            # Get intraday data
            intraday_data = self.market_data.get_intraday_data(
                stock.symbol, interval='1m', lookback_days=1
            )
            
            if intraday_data.empty:
                continue
                
            # Store price history in the stock object
            stock.set_price_history(intraday_data)
            
            # Check for bull flag pattern
            bull_flags = self.pattern_detector.detect_bull_flag(intraday_data)
            if bull_flags:
                stock.has_bull_flag = True
                self.bull_flags[stock.symbol] = stock
                self.tracked_stocks[stock.symbol] = 'bull_flag'
                logger.info(f"Found bull flag pattern on {stock.symbol}")
            
            # Check for micro pullback pattern
            micro_pullbacks = self.pattern_detector.detect_micro_pullback(intraday_data)
            if micro_pullbacks:
                stock.has_micro_pullback = True
                self.micro_pullbacks[stock.symbol] = stock
                self.tracked_stocks[stock.symbol] = 'micro_pullback'
                logger.info(f"Found micro pullback pattern on {stock.symbol}")
            
            # Check for first candle to make a new high
            new_high_breakouts = self.pattern_detector.detect_first_candle_to_make_new_high(intraday_data)
            if new_high_breakouts:
                stock.has_new_high_breakout = True
                self.new_high_breakouts[stock.symbol] = stock
                self.tracked_stocks[stock.symbol] = 'new_high_breakout'
                logger.info(f"Found new high breakout pattern on {stock.symbol}")
        
        logger.info(f"Found {len(self.tracked_stocks)} stocks with active trading patterns")
        return self.tracked_stocks
    
    def get_actionable_stocks(self, max_stocks=5):
        """
        Get the top actionable stocks based on pattern priority.
        
        Parameters:
        -----------
        max_stocks: int
            Maximum number of stocks to return
            
        Returns:
        --------
        list: List of the top actionable stocks
        """
        # Prioritize patterns based on Ross Cameron's strategy
        # Priority: 1. Bull Flags, 2. New High Breakouts, 3. Micro Pullbacks
        actionable_stocks = []
        
        # Add stocks with bull flags
        for symbol, stock in self.bull_flags.items():
            if len(actionable_stocks) >= max_stocks:
                break
            actionable_stocks.append(stock)
        
        # Add stocks with new high breakouts
        if len(actionable_stocks) < max_stocks:
            for symbol, stock in self.new_high_breakouts.items():
                if symbol in [s.symbol for s in actionable_stocks]:
                    continue
                if len(actionable_stocks) >= max_stocks:
                    break
                actionable_stocks.append(stock)
        
        # Add stocks with micro pullbacks
        if len(actionable_stocks) < max_stocks:
            for symbol, stock in self.micro_pullbacks.items():
                if symbol in [s.symbol for s in actionable_stocks]:
                    continue
                if len(actionable_stocks) >= max_stocks:
                    break
                actionable_stocks.append(stock)
        
        return actionable_stocks
    
    def generate_alerts(self):
        """
        Generate trading alerts for actionable stocks.
        
        Returns:
        --------
        list: List of alert dictionaries containing trading information
        """
        alerts = []
        
        # Get top actionable stocks
        actionable_stocks = self.get_actionable_stocks(max_stocks=5)
        
        for stock in actionable_stocks:
            # Determine pattern type
            pattern_type = self.tracked_stocks.get(stock.symbol, 'unknown')
            
            # Skip if we can't determine pattern type
            if pattern_type == 'unknown':
                continue
                
            # Get entry, stop, and target prices
            entry_price = stock.get_optimal_entry()
            stop_loss = stock.get_optimal_stop_loss()
            target = stock.get_optimal_target()
            
            # Skip if we can't determine prices
            if entry_price is None or stop_loss is None or target is None:
                continue
                
            # Create alert dictionary
            alert = {
                'timestamp': datetime.now(),
                'symbol': stock.symbol,
                'pattern': pattern_type,
                'price': stock.current_price,
                'entry': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk_per_share': entry_price - stop_loss,
                'reward_per_share': target - entry_price,
                'risk_reward_ratio': (target - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 0
            }
            
            # Add formatted message
            pattern_names = {
                'bull_flag': 'Bull Flag',
                'micro_pullback': 'Micro Pullback',
                'new_high_breakout': 'New High Breakout'
            }
            
            alert['message'] = (
                f"{pattern_names.get(pattern_type, 'Unknown Pattern')} ALERT: {stock.symbol} at "
                f"${stock.current_price:.2f} - Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}, "
                f"Target: ${target:.2f}"
            )
            
            alerts.append(alert)
            self.alert_history.append(alert)
            
            logger.info(f"Generated alert: {alert['message']}")
        
        return alerts
    
    def is_market_healthy(self):
        """
        Check if the overall market is healthy for day trading.
        Ross Cameron emphasizes the importance of trading only in favorable market conditions.
        
        Returns:
        --------
        bool: True if market conditions are favorable for day trading
        """
        # Update market conditions
        self.update_market_conditions()
        
        # Check if market is open
        self.update_market_session()
        
        # Market is healthy if it's open and strong
        is_healthy = (self.market_open or self.pre_market) and self.strong_market
        
        logger.info(f"Market health check: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        return is_healthy
    
    def get_consecutive_green_days(self):
        """
        Track consecutive green days to implement Ross Cameron's consistency strategy.
        In a real implementation, this would connect to a database of historical results.
        
        Returns:
        --------
        int: Number of consecutive green days
        """
        # Simulated implementation
        # In a real application, this would query a database or API
        return 1  # Placeholder
    
    def should_use_reduced_size(self, current_profit, daily_goal):
        """
        Determine if reduced position size should be used based on Ross Cameron's strategy.
        Use quarter size until reaching quarter of daily goal.
        
        Parameters:
        -----------
        current_profit: float
            Current profit for the day
        daily_goal: float
            Target profit for the day
            
        Returns:
        --------
        bool: True if reduced size should be used
        """
        quarter_goal = daily_goal * 0.25
        
        # Use reduced size until quarter goal is achieved
        use_reduced_size = current_profit < quarter_goal
        
        logger.info(f"Position size check: {'Using reduced size' if use_reduced_size else 'Using full size'}")
        
        return use_reduced_size