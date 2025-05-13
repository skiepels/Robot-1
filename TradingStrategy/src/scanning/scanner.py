"""
Scanner Module

Implements Ross Cameron's stock scanning criteria to find high-momentum stocks with:
1. 5x relative volume
2. Up 10% or more
3. Price between $1-$20
4. Float under 10 million shares 
5. Has a news catalyst
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.patterns import CandlestickPatterns

logger = logging.getLogger(__name__)


class StockScanner:
    def __init__(self, market_data_provider, news_provider):
        """
        Initialize the stock scanner.
        
        Parameters:
        -----------
        market_data_provider: MarketDataProvider
            Provider for market data (prices, volume, etc.)
        news_provider: NewsDataProvider
            Provider for news data
        """
        self.market_data = market_data_provider
        self.news = news_provider
        self.pattern_detector = CandlestickPatterns()
        
    def scan_for_momentum_stocks(self, min_price=1.0, max_price=20.0, 
                                min_gap_pct=10.0, min_rel_volume=5.0, 
                                max_float=10_000_000):
        """
        Scan the market for stocks meeting Ross Cameron's criteria:
        
        Parameters:
        -----------
        min_price: float
            Minimum stock price to consider
        max_price: float
            Maximum stock price to consider
        min_gap_pct: float
            Minimum gap up percentage
        min_rel_volume: float
            Minimum relative volume (compared to 50-day average)
        max_float: int
            Maximum shares in float
            
        Returns:
        --------
        list: Stock objects that meet the criteria
        """
        logger.info("Scanning for momentum stocks...")
        
        # Get all tradable stocks
        all_stocks = self.market_data.get_tradable_stocks()
        
        # Filter by price range
        filtered_by_price = [
            stock for stock in all_stocks 
            if min_price <= stock.current_price <= max_price
        ]
        
        logger.info(f"Found {len(filtered_by_price)} stocks in price range ${min_price}-${max_price}")
        
        # Filter by gap percentage
        filtered_by_gap = [
            stock for stock in filtered_by_price
            if stock.gap_percent >= min_gap_pct
        ]
        
        logger.info(f"Found {len(filtered_by_gap)} stocks gapping up {min_gap_pct}% or more")
        
        # Filter by relative volume
        filtered_by_volume = [
            stock for stock in filtered_by_gap
            if stock.relative_volume >= min_rel_volume
        ]
        
        logger.info(f"Found {len(filtered_by_volume)} stocks with {min_rel_volume}x or higher relative volume")
        
        # Filter by float size
        filtered_by_float = [
            stock for stock in filtered_by_volume
            if stock.shares_float <= max_float
        ]
        
        logger.info(f"Found {len(filtered_by_float)} stocks with float under {max_float:,} shares")
        
        # Filter by news catalyst
        stocks_with_catalysts = []
        for stock in filtered_by_float:
            news_items = self.news.get_stock_news(stock.symbol, days=1)
            if news_items:
                stock.has_news = True
                stock.news_headline = news_items[0].headline
                stock.news_source = news_items[0].source
                stock.news_timestamp = news_items[0].date
                stocks_with_catalysts.append(stock)
        
        logger.info(f"Found {len(stocks_with_catalysts)} stocks with news catalysts")
        
        # Sort by overall strength score
        sorted_stocks = self._sort_by_strength(stocks_with_catalysts)
        
        return sorted_stocks
    
    def scan_for_bull_flags(self, candidates, lookback_period=10):
        """
        Scan for bull flag patterns in the candidate stocks.
        
        Parameters:
        -----------
        candidates: list
            List of Stock objects to scan
        lookback_period: int
            Number of periods to look back for pattern formation
            
        Returns:
        --------
        list: Stock objects with bull flag patterns, sorted by strength
        """
        bull_flags = []
        
        for stock in candidates:
            # Get recent price data
            ohlcv = self.market_data.get_intraday_data(
                stock.symbol, 
                interval='1m', 
                lookback_days=1
            )
            
            if ohlcv.empty:
                continue
                
            # Store price history in the stock object
            stock.set_price_history(ohlcv)
            
            # Check for bull flag pattern
            flags = self.pattern_detector.detect_bull_flag(ohlcv, lookback=lookback_period)
            
            if flags:
                stock.has_bull_flag = True
                bull_flags.append(stock)
                logger.info(f"Found bull flag pattern on {stock.symbol}")
                
        logger.info(f"Found {len(bull_flags)} stocks with bull flag patterns")
        
        # Sort by strength of bull flag
        sorted_flags = self._sort_bull_flags_by_strength(bull_flags)
        
        return sorted_flags
    
    def scan_for_first_candle_to_make_new_high(self, candidates, lookback_period=5):
        """
        Scan for the "first candle to make a new high" pattern in candidate stocks.
        This is Ross Cameron's key entry setup.
        
        Parameters:
        -----------
        candidates: list
            List of Stock objects to scan
        lookback_period: int
            Number of periods to look back for pattern formation
            
        Returns:
        --------
        list: Stock objects with the pattern, sorted by strength
        """
        new_high_candidates = []
        
        for stock in candidates:
            # Get recent price data
            ohlcv = self.market_data.get_intraday_data(
                stock.symbol, 
                interval='1m', 
                lookback_days=1
            )
            
            if ohlcv.empty:
                continue
                
            # Store price history in the stock object if not already set
            if stock.price_history is None:
                stock.set_price_history(ohlcv)
            
            # Check for "first candle to make a new high" pattern
            new_highs = self.pattern_detector.detect_first_candle_to_make_new_high(
                ohlcv, lookback=lookback_period
            )
            
            if new_highs:
                stock.has_new_high_breakout = True
                new_high_candidates.append(stock)
                logger.info(f"Found first candle to make new high pattern on {stock.symbol}")
                
        logger.info(f"Found {len(new_high_candidates)} stocks with first candle making new high")
        
        # Sort by strength of the pattern
        sorted_candidates = self._sort_by_new_high_strength(new_high_candidates)
        
        return sorted_candidates
    
    def scan_for_micro_pullbacks(self, candidates, lookback_period=3):
        """
        Scan for micro pullback patterns in the candidate stocks.
        
        Parameters:
        -----------
        candidates: list
            List of Stock objects to scan
        lookback_period: int
            Number of periods to look back for pattern formation
            
        Returns:
        --------
        list: Stock objects with micro pullback patterns, sorted by strength
        """
        micro_pullbacks = []
        
        for stock in candidates:
            # Get recent price data
            ohlcv = self.market_data.get_intraday_data(
                stock.symbol, 
                interval='1m', 
                lookback_days=1
            )
            
            if ohlcv.empty:
                continue
                
            # Store price history in the stock object if not already set
            if stock.price_history is None:
                stock.set_price_history(ohlcv)
            
            # Check for micro pullback pattern
            pullbacks = self.pattern_detector.detect_micro_pullback(
                ohlcv, lookback=lookback_period
            )
            
            if pullbacks:
                stock.has_micro_pullback = True
                micro_pullbacks.append(stock)
                logger.info(f"Found micro pullback pattern on {stock.symbol}")
                
        logger.info(f"Found {len(micro_pullbacks)} stocks with micro pullback patterns")
        
        # Sort by strength of the pattern
        sorted_pullbacks = self._sort_micro_pullbacks_by_strength(micro_pullbacks)
        
        return sorted_pullbacks
    
    def _sort_by_strength(self, stocks):
        """
        Sort stocks by overall momentum strength score.
        
        Parameters:
        -----------
        stocks: list
            List of Stock objects to sort
            
        Returns:
        --------
        list: Sorted list of Stock objects
        """
        if not stocks:
            return []
            
        # Calculate a composite strength score for each stock
        strength_scores = {}
        
        for stock in stocks:
            # Percentage change score (0-10)
            pct_change_score = min(stock.gap_percent / 5, 10)
            
            # Relative volume score (0-10)
            rel_vol_score = min(stock.relative_volume / 2, 10)
            
            # Float score (0-10, lower float is better)
            if stock.shares_float > 0:
                float_score = max(10 - (stock.shares_float / 1_000_000), 0)
            else:
                float_score = 0
            
            # News impact score (0-10)
            news_score = 0
            news_items = self.news.get_stock_news(stock.symbol, days=1)
            if news_items:
                # Use the highest impact news item
                highest_impact = max([item.score for item in news_items], default=0)
                news_score = min(highest_impact, 10)
            
            # Calculate total strength score
            total_score = pct_change_score + rel_vol_score + float_score + news_score
            strength_scores[stock.symbol] = total_score
        
        # Sort stocks by descending strength score
        sorted_stocks = sorted(
            stocks, 
            key=lambda x: strength_scores.get(x.symbol, 0), 
            reverse=True
        )
        
        return sorted_stocks
    
    def _sort_bull_flags_by_strength(self, stocks):
        """
        Sort stocks with bull flag patterns by pattern strength.
        
        Parameters:
        -----------
        stocks: list
            List of Stock objects with bull flag patterns
            
        Returns:
        --------
        list: Sorted list of Stock objects
        """
        if not stocks:
            return []
            
        # Calculate pattern strength for each stock
        pattern_scores = {}
        
        for stock in stocks:
            # Make sure we have price history
            if stock.price_history is None or stock.price_history.empty:
                pattern_scores[stock.symbol] = 0
                continue
                
            ohlcv = stock.price_history
            
            # Calculate pole strength (percent increase)
            pole_start_idx = max(0, len(ohlcv) - 15)
            pole_end_idx = max(0, len(ohlcv) - 7)
            
            if pole_start_idx >= pole_end_idx:
                pattern_scores[stock.symbol] = 0
                continue
                
            pole_start_price = ohlcv['close'].iloc[pole_start_idx]
            pole_end_price = ohlcv['close'].iloc[pole_end_idx]
            
            pole_strength = (pole_end_price / pole_start_price - 1) * 100
            
            # Calculate flag quality (lower range is better)
            flag_start_idx = pole_end_idx
            flag_end_idx = len(ohlcv) - 1
            
            if flag_start_idx >= flag_end_idx:
                pattern_scores[stock.symbol] = 0
                continue
                
            flag_data = ohlcv.iloc[flag_start_idx:flag_end_idx+1]
            
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            
            flag_range_pct = (flag_high - flag_low) / pole_end_price * 100
            
            # Quality increases as flag range decreases
            flag_quality = max(10 - flag_range_pct, 0)
            
            # Volume pattern (should decrease during flag)
            if 'volume' in ohlcv.columns and len(flag_data) >= 4:
                first_half_vol = flag_data['volume'].iloc[:len(flag_data)//2].mean()
                second_half_vol = flag_data['volume'].iloc[len(flag_data)//2:].mean()
                
                # Higher score if volume is decreasing
                vol_pattern_score = 5 if second_half_vol < first_half_vol else 0
            else:
                vol_pattern_score = 0
            
            # Calculate total pattern strength
            total_score = pole_strength + flag_quality + vol_pattern_score
            pattern_scores[stock.symbol] = total_score
        
        # Sort stocks by descending pattern strength
        sorted_stocks = sorted(
            stocks, 
            key=lambda x: pattern_scores.get(x.symbol, 0), 
            reverse=True
        )
        
        return sorted_stocks
    
    def _sort_by_new_high_strength(self, stocks):
        """
        Sort stocks with "first candle to make a new high" patterns by pattern strength.
        
        Parameters:
        -----------
        stocks: list
            List of Stock objects with new high patterns
            
        Returns:
        --------
        list: Sorted list of Stock objects
        """
        if not stocks:
            return []
            
        # Calculate pattern strength for each stock
        pattern_scores = {}
        
        for stock in stocks:
            # Make sure we have price history
            if stock.price_history is None or stock.price_history.empty:
                pattern_scores[stock.symbol] = 0
                continue
                
            ohlcv = stock.price_history
            
            # Get the breakout candle
            breakout_idx = len(ohlcv) - 1
            
            # Check if we have enough data
            if breakout_idx < 1:
                pattern_scores[stock.symbol] = 0
                continue
                
            breakout_candle = ohlcv.iloc[breakout_idx]
            prev_candle = ohlcv.iloc[breakout_idx-1]
            
            # Calculate strength factors
            
            # 1. Breakout volume strength
            if 'volume' in ohlcv.columns:
                breakout_vol = breakout_candle['volume']
                avg_vol = ohlcv['volume'].iloc[max(0, breakout_idx-5):breakout_idx].mean()
                
                vol_strength = min(breakout_vol / max(avg_vol, 1), 5) * 2
            else:
                vol_strength = 0
            
            # 2. Breakout candle body strength
            if breakout_candle['high'] > breakout_candle['low']:
                body_ratio = abs(breakout_candle['close'] - breakout_candle['open']) / (breakout_candle['high'] - breakout_candle['low'])
                body_strength = body_ratio * 5
            else:
                body_strength = 0
            
            # 3. Breakout above previous candle
            if prev_candle['high'] > 0:
                breakout_pct = (breakout_candle['high'] / prev_candle['high'] - 1) * 100
                breakout_strength = min(breakout_pct * 2, 5)
            else:
                breakout_strength = 0
            
            # Calculate total pattern strength
            total_score = vol_strength + body_strength + breakout_strength
            pattern_scores[stock.symbol] = total_score
        
        # Sort stocks by descending pattern strength
        sorted_stocks = sorted(
            stocks, 
            key=lambda x: pattern_scores.get(x.symbol, 0), 
            reverse=True
        )
        
        return sorted_stocks
    
    def _sort_micro_pullbacks_by_strength(self, stocks):
        """
        Sort stocks with micro pullback patterns by pattern strength.
        
        Parameters:
        -----------
        stocks: list
            List of Stock objects with micro pullback patterns
            
        Returns:
        --------
        list: Sorted list of Stock objects
        """
        if not stocks:
            return []
            
        # Calculate pattern strength for each stock
        pattern_scores = {}
        
        for stock in stocks:
            # Make sure we have price history
            if stock.price_history is None or stock.price_history.empty:
                pattern_scores[stock.symbol] = 0
                continue
                
            ohlcv = stock.price_history
            
            # Get the pullback candle and the candle after it
            pullback_idx = len(ohlcv) - 2
            breakout_idx = len(ohlcv) - 1
            
            # Check if we have enough data
            if pullback_idx < 1 or breakout_idx >= len(ohlcv):
                pattern_scores[stock.symbol] = 0
                continue
                
            pullback_candle = ohlcv.iloc[pullback_idx]
            breakout_candle = ohlcv.iloc[breakout_idx]
            
            # Calculate strength factors
            
            # 1. Pullback depth (smaller is better)
            prior_candles = ohlcv.iloc[max(0, pullback_idx-3):pullback_idx]
            if not prior_candles.empty:
                prior_high = prior_candles['high'].max()
                pullback_low = pullback_candle['low']
                
                if prior_high > pullback_low:
                    pullback_depth = (prior_high - pullback_low) / prior_high * 100
                    depth_score = max(5 - pullback_depth, 0)
                else:
                    depth_score = 0
            else:
                depth_score = 0
            
            # 2. Breakout strength
            if pullback_candle['high'] > 0:
                breakout_pct = (breakout_candle['high'] / pullback_candle['high'] - 1) * 100
                breakout_strength = min(breakout_pct * 2, 5)
            else:
                breakout_strength = 0
            
            # 3. Volume on breakout
            if 'volume' in ohlcv.columns:
                breakout_vol = breakout_candle['volume']
                pullback_vol = pullback_candle['volume']
                
                if pullback_vol > 0:
                    vol_increase = breakout_vol / pullback_vol
                    vol_score = min(vol_increase, 5)
                else:
                    vol_score = 0
            else:
                vol_score = 0
            
            # Calculate total pattern strength
            total_score = depth_score + breakout_strength + vol_score
            pattern_scores[stock.symbol] = total_score
        
        # Sort stocks by descending pattern strength
        sorted_stocks = sorted(
            stocks, 
            key=lambda x: pattern_scores.get(x.symbol, 0), 
            reverse=True
        )
        
        return sorted_stocks