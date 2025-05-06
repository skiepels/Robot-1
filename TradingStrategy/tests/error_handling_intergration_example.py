"""
Error Handling Integration Example

This script demonstrates how to integrate the error handling utilities
into the existing trading strategy code.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling utilities
from error_handler import exception_handler, timeout_handler, measure_performance, ErrorHandler

# Import trading strategy components
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.scanning.scanner import StockScanner
from src.scanning.condition_tracker import ConditionTracker
from src.trading.risk_manager import RiskManager
from src.trading.trade_manager import TradeManager
from src.utils.logger import setup_logger


# Setup logger
logger = setup_logger('error_integration', log_dir='logs', console_level=logging.INFO)

# Create error handler
error_handler = ErrorHandler(log_dir='logs')


# Example 1: Add error handling to market data provider methods
class EnhancedMarketDataProvider(MarketDataProvider):
    """Enhanced market data provider with error handling."""
    
    @exception_handler(retries=3, retry_delay=2, fallback_return=pd.DataFrame())
    @measure_performance(threshold_ms=500)
    def get_intraday_data(self, symbol, interval='1m', lookback_days=1):
        """
        Get intraday price data for a stock with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().get_intraday_data(symbol, interval, lookback_days)
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='get_intraday_data',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise
    
    @exception_handler(retries=2, retry_delay=1, fallback_return=None)
    @measure_performance(threshold_ms=200)
    def get_current_price(self, symbol):
        """
        Get the current market price for a stock with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().get_current_price(symbol)
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='get_current_price',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise


# Example 2: Add error handling to candlestick pattern detection
class EnhancedCandlestickPatterns(CandlestickPatterns):
    """Enhanced candlestick pattern detection with error handling."""
    
    @exception_handler(fallback_return=[])
    @measure_performance(threshold_ms=300)
    def detect_bull_flag(self, df, lookback=7):
        """
        Detect a Bull Flag pattern with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().detect_bull_flag(df, lookback)
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='detect_bull_flag',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise
    
    @exception_handler(fallback_return=[])
    @measure_performance(threshold_ms=300)
    def detect_micro_pullback(self, df, lookback=3):
        """
        Detect Micro Pullback patterns with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().detect_micro_pullback(df, lookback)
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='detect_micro_pullback',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise


# Example 3: Add error handling to trade manager
class EnhancedTradeManager(TradeManager):
    """Enhanced trade manager with error handling."""
    
    @exception_handler(fallback_return=[])
    @measure_performance(threshold_ms=1000)
    def scan_for_opportunities(self):
        """
        Scan for trading opportunities with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().scan_for_opportunities()
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='scan_for_opportunities',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise
    
    @exception_handler
    @timeout_handler(timeout=5, fallback_return=None)
    @measure_performance(threshold_ms=500)
    def execute_trade(self, trade_params):
        """
        Execute a trade with error handling and timeout protection.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().execute_trade(trade_params)
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='execute_trade',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise
    
    @exception_handler(retries=2, fallback_return=[])
    @measure_performance(threshold_ms=500)
    def manage_active_trades(self):
        """
        Manage active trades with error handling.
        
        This overrides the base method to add error handling capabilities.
        """
        try:
            return super().manage_active_trades()
        except Exception as e:
            # Log the specific error
            error_handler.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                module=self.__class__.__name__,
                function='manage_active_trades',
                severity='ERROR'
            )
            # Re-raise to let the decorator handle it
            raise


# Example application - A simple trading loop with error handling
def run_trading_loop_with_error_handling():
    """Run a simple trading loop with enhanced error handling."""
    logger.info("Starting trading loop with enhanced error handling")
    
    try:
        # Initialize enhanced components
        market_data = EnhancedMarketDataProvider()
        news_data = NewsDataProvider()
        
        # Use pattern detector with enhanced error handling
        pattern_detector = EnhancedCandlestickPatterns()
        
        # Create scanner and condition tracker
        scanner = StockScanner(market_data, news_data)
        condition_tracker = ConditionTracker(market_data, news_data)
        
        # Override the pattern detector in condition tracker
        condition_tracker.pattern_detector = pattern_detector
        
        # Create risk manager
        risk_manager = RiskManager(
            initial_capital=10000.0,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Create enhanced trade manager
        trade_manager = EnhancedTradeManager(
            market_data_provider=market_data,
            scanner=scanner,
            risk_manager=risk_manager,
            condition_tracker=condition_tracker,
            broker_api=None  # Simulation mode
        )
        
        # Set trading parameters
        trade_manager.min_price = 1.0
        trade_manager.max_price = 20.0
        trade_manager.min_gap_pct = 10.0
        trade_manager.min_rel_volume = 5.0
        trade_manager.max_float = 10_000_000
        
        # Main trading loop (run for a limited time for testing)
        logger.info("Starting main trading loop")
        end_time = datetime.now() + timedelta(minutes=5)
        
        while datetime.now() < end_time:
            # Update market conditions
            try:
                condition_tracker.update_market_conditions()
            except Exception as e:
                error_handler.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    module='run_trading_loop',
                    function='market_conditions_update',
                    severity='ERROR'
                )
                # Continue despite error
                logger.warning("Continuing after error in market conditions update")
            
            # Scan for opportunities
            opportunities = trade_manager.scan_for_opportunities()
            
            if opportunities:
                logger.info(f"Found {len(opportunities)} trading opportunities")
                
                # Process each opportunity
                for stock in opportunities:
                    try:
                        # Evaluate the opportunity
                        trade_params = trade_manager.evaluate_opportunity(stock)
                        
                        if trade_params:
                            # Execute trade
                            executed_trade = trade_manager.execute_trade(trade_params)
                            
                            if executed_trade:
                                logger.info(f"Executed trade: {stock.symbol}")
                    except Exception as e:
                        error_handler.log_error(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            traceback=traceback.format_exc(),
                            module='run_trading_loop',
                            function='process_opportunity',
                            severity='ERROR'
                        )
                        # Skip this opportunity but continue loop
                        logger.warning(f"Skipping opportunity for {stock.symbol} after error")
            
            # Manage active trades
            try:
                actions = trade_manager.manage_active_trades()
                
                if actions:
                    logger.info(f"Performed {len(actions)} trade management actions")
            except Exception as e:
                error_handler.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    module='run_trading_loop',
                    function='manage_trades',
                    severity='ERROR'
                )
                logger.warning("Error in trade management")
            
            # Wait before next update
            time.sleep(10)
        
        logger.info("Trading loop completed")
        
        # Generate error report
        report = error_handler.generate_error_report(
            output_file='logs/trading_error_report.txt'
        )
        
        logger.info("Error report generated")
        
    except Exception as e:
        error_handler.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            module='run_trading_loop',
            function='main',
            severity='CRITICAL'
        )
        logger.critical("Fatal error in trading loop")


if __name__ == '__main__':
    # Import missing modules for the example
    import time
    import traceback
    import pandas as pd
    from src.analysis.candlestick_patterns import CandlestickPatterns
    
    # Run the example
    run_trading_loop_with_error_handling()