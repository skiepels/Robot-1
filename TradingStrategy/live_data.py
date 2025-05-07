"""
Live Trading with Interactive Brokers

This script runs the Ross Cameron day trading strategy with live
data from Interactive Brokers for trading GPUS.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.scanning.scanner import StockScanner
from src.scanning.condition_tracker import ConditionTracker
from src.trading.risk_manager import RiskManager
from src.trading.trade_manager import TradeManager
from real_market_data import RealMarketDataProvider
from simple_news_provider import SimpleNewsProvider
from ib_broker import IBBroker

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run live trading with Interactive Brokers')
    
    parser.add_argument('--api-key', type=str, required=True,
                      help='Alpha Vantage API key (for market data)')
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='TWS/IB Gateway host')
    
    parser.add_argument('--port', type=int, default=7497,
                      help='TWS/IB Gateway port (7497 for paper, 7496 for live)')
    
    parser.add_argument('--client-id', type=int, default=1,
                      help='Client ID for IB API connection')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital')
    
    parser.add_argument('--paper', action='store_true',
                      help='Use paper trading (safe default)')
    
    return parser.parse_args()

def run_live_trading(args):
    """
    Run live trading with Interactive Brokers.
    
    Parameters:
    -----------
    args: argparse.Namespace
        Command line arguments
    """
    # Setup logger
    logger = setup_logger('live_trading', log_dir='logs', console_level=logging.INFO)
    
    logger.info("Starting live trading with Interactive Brokers")
    
    # Determine if using paper trading
    is_paper = args.paper or args.port == 7497
    trading_mode = "PAPER" if is_paper else "LIVE"
    
    logger.info(f"Trading mode: {trading_mode}")
    
    if not is_paper:
        # Additional confirmation for live trading
        confirm = input("You are about to start LIVE trading with real money. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return
    
    try:
        # Initialize Interactive Brokers connection
        broker = IBBroker(
            host=args.host,
            port=args.port,
            client_id=args.client_id
        )
        
        # Connect to IB
        if not broker.connect():
            logger.error("Failed to connect to Interactive Brokers. Exiting.")
            return
        
        # Initialize other components
        market_data = RealMarketDataProvider(api_key=args.api_key)
        news_data = SimpleNewsProvider()
        
        # Initialize scanner
        scanner = StockScanner(market_data, news_data)
        
        # Initialize condition tracker
        condition_tracker = ConditionTracker(market_data, news_data)
        
        # Get actual account balance
        account_balance = broker.get_account_balance()
        logger.info(f"Account balance: ${account_balance:.2f}")
        
        # Initialize risk manager with actual balance
        risk_manager = RiskManager(
            initial_capital=account_balance,
            max_risk_per_trade_pct=1.0,
            daily_max_loss_pct=3.0,
            profit_loss_ratio=2.0,
            max_open_positions=3
        )
        
        # Initialize trade manager
        trade_manager = TradeManager(
            market_data_provider=market_data,
            scanner=scanner,
            risk_manager=risk_manager,
            condition_tracker=condition_tracker,
            broker_api=broker
        )
        
        # Set trading parameters
        trade_manager.min_price = 1.0
        trade_manager.max_price = 100.0
        trade_manager.min_gap_pct = 2.0
        trade_manager.min_rel_volume = 1.0
        trade_manager.max_float = 100_000_000
        
        # Start trading session
        trade_manager.start_trading_session()
        trade_manager.is_trading_enabled = True
        trade_manager.is_simulated = False
        
        logger.info("Trading session started")
        
        # Main trading loop
        try:
            while True:
                # Check market session
                condition_tracker.update_market_session()
                
                if not condition_tracker.market_open:
                    logger.info("Market is closed. Waiting for market open.")
                    time.sleep(60)
                    continue
                
                # Update market conditions
                condition_tracker.update_market_conditions()
                
                if not condition_tracker.is_market_healthy():
                    logger.warning("Market conditions unfavorable. Monitoring only.")
                    time.sleep(60)
                    continue
                
                # Scan for opportunities (focusing on GPUS)
                logger.info("Scanning for trading opportunities in GPUS...")
                
                # Get GPUS data
                stocks = market_data.get_tradable_stocks()  # Should contain GPUS
                
                if not stocks:
                    logger.warning("Failed to get data for GPUS. Retrying in 60 seconds.")
                    time.sleep(60)
                    continue
                
                # Scan for trading conditions
                tracked_stocks = condition_tracker.scan_for_trading_conditions(
                    stocks,
                    min_gap_pct=trade_manager.min_gap_pct,
                    min_rel_volume=trade_manager.min_rel_volume,
                    max_float=trade_manager.max_float
                )
                
                if tracked_stocks:
                    logger.info(f"Found trading opportunities: {list(tracked_stocks.keys())}")
                    
                    # Generate alerts
                    alerts = condition_tracker.generate_alerts()
                    
                    for alert in alerts:
                        logger.info(f"ALERT: {alert['message']}")
                        
                        # Get stock object
                        symbol = alert['symbol']
                        stock = next((s for s in stocks if s.symbol == symbol), None)
                        
                        if stock:
                            # Evaluate opportunity
                            opportunity = trade_manager.evaluate_opportunity(stock)
                            
                            if opportunity:
                                logger.info(f"Trade opportunity validated for {symbol}")
                                
                                # Execute trade
                                trade = trade_manager.execute_trade(opportunity)
                                
                                if trade:
                                    logger.info(f"Trade executed: {trade['symbol']} - {trade['executed_shares']} shares at ${trade['executed_price']:.2f}")
                            else:
                                logger.info(f"Trade opportunity not valid for {symbol}")
                else:
                    logger.info("No trading opportunities found")
                
                # Manage active trades
                actions = trade_manager.manage_active_trades()
                
                if actions:
                    logger.info(f"Performed {len(actions)} trade management actions")
                
                # Wait before next scan
                logger.info("Waiting for next scan cycle...")
                time.sleep(300)  # 5 minutes between scans
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        
        # Stop trading session
        trade_manager.stop_trading_session()
        
        # Disconnect from IB
        broker.disconnect()
        
        logger.info("Trading session ended")
        
    except Exception as e:
        logger.error(f"Error in live trading: {e}", exc_info=True)

if __name__ == '__main__':
    args = parse_arguments()
    run_live_trading(args)