"""
Stock Data Model

Implements a data model for storing and managing stock information
used by the day trading strategy based on Ross Cameron's approach.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Stock:
    def __init__(self, symbol, name=None):
        """
        Initialize a stock object.
        
        Parameters:
        -----------
        symbol: str
            Stock ticker symbol
        name: str
            Company name
        """
        self.symbol = symbol
        self.name = name or symbol
        
        # Current market data
        self.current_price = 0.0
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = 0.0
        self.previous_close = 0.0
        
        # Volume data
        self.current_volume = 0
        self.avg_volume_50d = 0
        self.relative_volume = 0.0
        
        # Share data
        self.shares_outstanding = 0
        self.shares_float = 0
        self.shares_short = 0
        self.short_ratio = 0.0
        
        # Price change data
        self.change_today = 0.0
        self.change_today_percent = 0.0
        self.gap_percent = 0.0
        
        # Technical indicators
        self.vwap = 0.0
        self.sma_20 = 0.0
        self.sma_50 = 0.0
        self.sma_200 = 0.0
        self.ema_9 = 0.0
        
        # News and catalyst data
        self.has_news = False
        self.news_headline = ""
        self.news_source = ""
        self.news_timestamp = None
        
        # Pattern identification
        self.has_bull_flag = False
        self.has_micro_pullback = False
        self.has_new_high_breakout = False
        
        # Trade data
        self.price_history = None  # DataFrame for storing historical price data
        self.current_pattern = None  # Current detected pattern
        
    def update_price(self, current_price, high_price=None, low_price=None, open_price=None):
        """Update current price and related fields"""
        self.current_price = current_price
        
        if high_price is not None:
            self.high_price = high_price
        elif current_price > self.high_price:
            self.high_price = current_price
            
        if low_price is not None:
            self.low_price = low_price
        elif self.low_price == 0 or current_price < self.low_price:
            self.low_price = current_price
            
        if open_price is not None:
            self.open_price = open_price
            
        # Update change calculations
        if self.previous_close > 0:
            self.change_today = self.current_price - self.previous_close
            self.change_today_percent = (self.change_today / self.previous_close) * 100
            
        if self.open_price > 0 and self.previous_close > 0:
            self.gap_percent = (self.open_price - self.previous_close) / self.previous_close * 100
    
    def update_volume(self, current_volume, avg_volume_50d=None):
        """Update volume data"""
        self.current_volume = current_volume
        
        if avg_volume_50d is not None:
            self.avg_volume_50d = avg_volume_50d
            
        if self.avg_volume_50d > 0:
            self.relative_volume = self.current_volume / self.avg_volume_50d
    
    def update_share_data(self, shares_outstanding=None, shares_float=None, 
                         shares_short=None, short_ratio=None):
        """Update share structure data"""
        if shares_outstanding is not None:
            self.shares_outstanding = shares_outstanding
            
        if shares_float is not None:
            self.shares_float = shares_float
            
        if shares_short is not None:
            self.shares_short = shares_short
            
        if short_ratio is not None:
            self.short_ratio = short_ratio
    
    def update_technical_indicators(self, vwap=None, sma_20=None, sma_50=None, 
                                  sma_200=None, ema_9=None):
        """Update technical indicators"""
        if vwap is not None:
            self.vwap = vwap
            
        if sma_20 is not None:
            self.sma_20 = sma_20
            
        if sma_50 is not None:
            self.sma_50 = sma_50
            
        if sma_200 is not None:
            self.sma_200 = sma_200
            
        if ema_9 is not None:
            self.ema_9 = ema_9
    
    def add_news(self, headline, source, timestamp=None):
        """Add news catalyst information"""
        self.has_news = True
        self.news_headline = headline
        self.news_source = source
        self.news_timestamp = timestamp or datetime.now()
    
    def set_price_history(self, df):
        """Set historical price data"""
        self.price_history = df
    
    def meets_criteria(self, min_price=1.0, max_price=20.0, min_gap_pct=10.0, 
                      min_rel_volume=5.0, max_float=10_000_000):
        """
        Check if stock meets Ross Cameron's criteria:
        1. Price between $1-$20
        2. Gapping up at least 10%
        3. Relative volume at least 5x
        4. Float under 10 million shares
        5. Has news (checked separately)
        """
        price_ok = min_price <= self.current_price <= max_price
        gap_ok = self.gap_percent >= min_gap_pct
        volume_ok = self.relative_volume >= min_rel_volume
        float_ok = self.shares_float <= max_float
        
        all_criteria_met = price_ok and gap_ok and volume_ok and float_ok
        
        return all_criteria_met
    
    def __str__(self):
        """String representation of the stock"""
        return (f"{self.symbol} - ${self.current_price:.2f} ({self.change_today_percent:.2f}%) | "
                f"Vol: {self.current_volume:,} ({self.relative_volume:.2f}x) | Float: {self.shares_float:,}")
    
    def get_risk_level(self):
        """
        Calculate risk level based on volatility and price.
        Returns a value from 1-10 with 10 being highest risk.
        """
        # Higher risk for very low-priced stocks
        price_risk = 10 - min(self.current_price, 10)
        
        # Higher risk for extremely high relative volume
        volume_risk = min(self.relative_volume / 2, 10)
        
        # Higher risk for smaller floats
        float_in_millions = self.shares_float / 1_000_000
        float_risk = max(10 - float_in_millions, 0)
        
        # Calculate average risk score
        avg_risk = (price_risk + volume_risk + float_risk) / 3
        
        return round(avg_risk, 1)
    
    def get_optimal_entry(self):
        """
        Based on the current pattern, determine the optimal entry price
        using Ross Cameron's strategy
        """
        if not self.price_history is not None or len(self.price_history) < 5:
            return None
            
        # Get recent data
        recent_data = self.price_history.iloc[-5:]
        
        if self.has_bull_flag:
            # For bull flag, entry is the first candle to make a new high after the flag
            flag_high = recent_data['high'].max()
            return flag_high
            
        elif self.has_micro_pullback:
            # For micro pullback, entry is above the current candle's high
            return recent_data.iloc[-1]['high']
            
        elif self.has_new_high_breakout:
            # For breakout, entry is the breakout level
            return recent_data.iloc[-1]['high']
            
        # Default to current price if no pattern detected
        return self.current_price
    
    def get_optimal_stop_loss(self):
        """
        Based on the current pattern, determine the optimal stop loss price
        using Ross Cameron's strategy
        """
        if not self.price_history is not None or len(self.price_history) < 5:
            return None
            
        # Get recent data
        recent_data = self.price_history.iloc[-5:]
        
        if self.has_bull_flag:
            # For bull flag, stop is below the low of the flag
            flag_low = recent_data['low'].min()
            return flag_low
            
        elif self.has_micro_pullback:
            # For micro pullback, stop is below the pullback low
            return recent_data.iloc[-2]['low']  # Pullback candle
            
        elif self.has_new_high_breakout:
            # For breakout, stop is below the breakout candle's low
            return recent_data.iloc[-1]['low']
            
        # Default to 5% below current price
        return self.current_price * 0.95
    
    def get_optimal_target(self):
        """
        Based on the current pattern, determine the optimal profit target
        using Ross Cameron's 2:1 profit-to-loss ratio
        """
        entry = self.get_optimal_entry()
        stop = self.get_optimal_stop_loss()
        
        if entry is None or stop is None:
            return None
            
        risk = entry - stop
        target = entry + (risk * 2)  # 2:1 ratio
        
        return target