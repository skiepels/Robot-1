"""
Base Pattern Class

This module provides the abstract base class for all candlestick patterns.
All pattern implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BasePattern(ABC):
    """
    Abstract base class for all candlestick patterns.
    
    This class defines the interface that all pattern implementations must follow.
    """
    
    def __init__(self, name, pattern_type, min_candles_required):
        """
        Initialize the base pattern.
        
        Parameters:
        -----------
        name: str
            Name of the pattern (e.g., "Hammer", "Bull Flag")
        pattern_type: str
            Type of pattern ("single", "double", "triple", "complex")
        min_candles_required: int
            Minimum number of candles needed to detect this pattern
        """
        self.name = name
        self.pattern_type = pattern_type
        self.min_candles_required = min_candles_required
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def detect(self, candles):
        """
        Detect the pattern in the provided candlestick data.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            DataFrame with OHLCV data. Must have columns: open, high, low, close, volume
            Index should be datetime
            
        Returns:
        --------
        dict or None: Pattern detection result with the following structure:
            {
                'pattern': str,          # Pattern name
                'confidence': float,     # Confidence score (0-100)
                'direction': str,        # 'bullish', 'bearish', or 'neutral'
                'entry_price': float,    # Suggested entry price
                'stop_price': float,     # Suggested stop loss price
                'target_price': float,   # Suggested target price (optional)
                'candle_index': int,     # Index where pattern was detected
                'notes': str            # Additional notes or observations
            }
            Returns None if pattern is not detected
        """
        pass
    
    def validate_candles(self, candles):
        """
        Validate that the candlestick data meets basic requirements.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data to validate
            
        Returns:
        --------
        bool: True if data is valid, False otherwise
        """
        # Check if candles is a DataFrame
        if not isinstance(candles, pd.DataFrame):
            self.logger.error("Candles must be a pandas DataFrame")
            return False
        
        # Check if we have enough candles
        if len(candles) < self.min_candles_required:
            self.logger.warning(f"Not enough candles. Need {self.min_candles_required}, got {len(candles)}")
            return False
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in candles.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        # Check for valid price data (no NaN values)
        if candles[required_columns].isnull().any().any():
            self.logger.error("Candle data contains NaN values")
            return False
        
        # Check for positive prices
        if (candles[required_columns] <= 0).any().any():
            self.logger.error("Candle data contains non-positive prices")
            return False
        
        return True
    
    def calculate_body_size(self, candle):
        """
        Calculate the body size of a candle.
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        float: Absolute body size
        """
        return abs(candle['close'] - candle['open'])
    
    def calculate_upper_wick(self, candle):
        """
        Calculate the upper wick size of a candle.
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        float: Upper wick size
        """
        body_top = max(candle['open'], candle['close'])
        return candle['high'] - body_top
    
    def calculate_lower_wick(self, candle):
        """
        Calculate the lower wick size of a candle.
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        float: Lower wick size
        """
        body_bottom = min(candle['open'], candle['close'])
        return body_bottom - candle['low']
    
    def calculate_range(self, candle):
        """
        Calculate the full range of a candle.
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        float: Full range (high - low)
        """
        return candle['high'] - candle['low']
    
    def is_bullish_candle(self, candle):
        """
        Determine if a candle is bullish (close > open).
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        bool: True if bullish, False otherwise
        """
        return candle['close'] > candle['open']
    
    def is_bearish_candle(self, candle):
        """
        Determine if a candle is bearish (close < open).
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
            
        Returns:
        --------
        bool: True if bearish, False otherwise
        """
        return candle['close'] < candle['open']
    
    def is_doji(self, candle, threshold=0.1):
        """
        Determine if a candle is a doji.
        
        Parameters:
        -----------
        candle: pandas.Series
            Single candle data
        threshold: float
            Body to range ratio threshold for doji classification
            
        Returns:
        --------
        bool: True if doji, False otherwise
        """
        body_size = self.calculate_body_size(candle)
        range_size = self.calculate_range(candle)
        
        if range_size == 0:
            return False
            
        return (body_size / range_size) <= threshold
    
    def calculate_trend(self, candles, lookback=20):
        """
        Calculate the trend direction based on recent candles.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data
        lookback: int
            Number of candles to look back for trend
            
        Returns:
        --------
        str: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(candles) < lookback:
            lookback = len(candles)
            
        recent_candles = candles.iloc[-lookback:]
        
        # Calculate linear regression on closing prices
        x = np.arange(len(recent_candles))
        y = recent_candles['close'].values
        
        # Fit linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Calculate slope as percentage of price
        avg_price = recent_candles['close'].mean()
        slope_pct = (slope / avg_price) * 100
        
        # Determine trend based on slope
        if slope_pct > 0.5:  # Uptrend if slope > 0.5% per candle
            return 'uptrend'
        elif slope_pct < -0.5:  # Downtrend if slope < -0.5% per candle
            return 'downtrend'
        else:
            return 'sideways'
    
    def calculate_volume_trend(self, candles, lookback=5):
        """
        Calculate if volume is increasing or decreasing.
        
        Parameters:
        -----------
        candles: pandas.DataFrame
            Candlestick data with volume
        lookback: int
            Number of candles to analyze
            
        Returns:
        --------
        str: 'increasing', 'decreasing', or 'stable'
        """
        if 'volume' not in candles.columns:
            return 'unknown'
            
        if len(candles) < lookback:
            lookback = len(candles)
            
        recent_volumes = candles['volume'].iloc[-lookback:].values
        
        # Calculate linear regression on volumes
        x = np.arange(len(recent_volumes))
        y = recent_volumes
        
        # Fit linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Calculate slope as percentage of average volume
        avg_volume = np.mean(recent_volumes)
        if avg_volume == 0:
            return 'stable'
            
        slope_pct = (slope / avg_volume) * 100
        
        # Determine volume trend
        if slope_pct > 10:  # Increasing if slope > 10% per candle
            return 'increasing'
        elif slope_pct < -10:  # Decreasing if slope < -10% per candle
            return 'decreasing'
        else:
            return 'stable'
    
    def calculate_confidence(self, pattern_score, trend_alignment=True, volume_confirmation=True):
        """
        Calculate confidence score for a pattern.
        
        Parameters:
        -----------
        pattern_score: float
            Base pattern score (0-100)
        trend_alignment: bool
            Whether pattern aligns with trend
        volume_confirmation: bool
            Whether volume confirms the pattern
            
        Returns:
        --------
        float: Confidence score (0-100)
        """
        confidence = pattern_score
        
        # Add bonus for trend alignment
        if trend_alignment:
            confidence = min(100, confidence + 10)
        else:
            confidence = max(0, confidence - 10)
        
        # Add bonus for volume confirmation
        if volume_confirmation:
            confidence = min(100, confidence + 10)
        else:
            confidence = max(0, confidence - 5)
        
        return confidence
    
    def log_detection(self, result):
        """
        Log pattern detection result.
        
        Parameters:
        -----------
        result: dict
            Pattern detection result
        """
        if result:
            self.logger.info(f"Detected {result['pattern']} pattern with {result['confidence']:.1f}% confidence")
        else:
            self.logger.debug(f"No {self.name} pattern detected")