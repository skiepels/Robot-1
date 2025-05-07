"""
Live Trading Script

This script runs the trading strategy with live data from Interactive Brokers.
It implements the 5 condition momentum trading strategy.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.data.ib_connector import IBConnector
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.scanning.scanner import StockScanner
from src.scanning.condition_tracker import ConditionTracker
from src.trading.risk_manager import RiskManager
from src.trading.trade_manager import TradeManager
from src.trading.ib_broker import IBBroker

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run live trading with Interactive Brokers')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=['GPUS'],
                      help='Stock symbols to trade (default: GPUS)')
    
    parser.add_argument('--paper', action='store_true',
                      help='Use paper trading (default)')
    
    parser.add_argument('--live', action='store_true',
                      help='Use live trading (overrides --paper)')
    
    parser.add_argument('--capital', type=float, default=None,
                      help='Initial capital (default: from .env)')
    
    parser.add_argument('--host', type=str, default=None,
                      help='TWS/IB Gateway host (default: from .env)')
    
    parser.add_argument('--port', type=int, default=None,
                      help='TWS/IB Gateway port (default: from .env)')
    
    parser.add_argument('--client-id', type=int, default=None,
                      help='Client ID for IB API connection (default: from .env)')
    
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
    
    # Determine trading mode
    is_paper = not args.live
    port = args.port or (int(os.getenv('IB_PORT', 7497)) if is_paper else 7496)
    trading_mode = "PAPER" if is_paper else "LIVE"
    
    logger.info(f"Trading mode: {trading_mode}")
    
    if not is_paper:
        # Additional confirmation for live trading
        confirm = input("You are about to start LIVE trading with real money. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return
    
    try:
        # Initialize IB connector
        ib_connector = IBConnector(
            host=args.host or os.getenv('IB_HOST'),
            port=port,
            client_id=args.client_id or int(os.getenv('IB_CLIENT_ID', 1))
        )
        
        if not ib_connector.connect():
            logger.error("Failed to connect to Interactive Brokers. Exiting.")
            return
        
        # Initialize IB broker
        ib_broker = IBBroker(connector=ib_connector)
        
        # Initialize market data provider
        market_data = MarketDataProvider(ib_connector=ib_connector)
        
        # Initialize news data provider
        news_data = NewsDataProvider()
        
        # Initialize scanner
        scanner = StockScanner(market_data, news_data)
        
        # Initialize condition tracker
        condition_tracker = ConditionTracker(market_data, news_data)
        
        # Get actual account balance
        account_balance = ib_broker.get_account_balance()
        logger.info(f"Account balance: ${account_balance:.2f}")
        
        # Use provided capital or account balance
        initial_capital = args.capital or account_balance or float(os.getenv('INITIAL_CAPITAL', 10000.0))
        
        # Initialize risk manager
        risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=float(os.getenv('MAX_RISK_PER_TRADE_PCT', 1.0)),
            daily_max_loss_pct=float(os.getenv('DAILY_MAX_LOSS_PCT', 3.0)),
            profit_loss_ratio=float(os.getenv('PROFIT_LOSS_RATIO', 2.0)),
            max_open_positions=int(os.getenv('MAX_OPEN_POSITIONS', 3))
        )
        
        # Initialize trade manager
        trade_manager = TradeManager(
            market_data_provider=market_data,
            scanner=scanner,
            risk_manager=risk_manager,
            condition_tracker=condition_tracker,
            broker_api=ib_broker
        )
        
        # Set trading parameters
        trade_manager.min_price = 2.0
        trade_manager.max_price = 20.0
        trade_manager.min_gap_pct = 10.0
        trade_manager.min_rel_volume = 5.0
        trade_manager.max_float = 20_000_000
        
        # Start trading session
        trade_manager.start_trading_session()
        trade_manager.is_trading_enabled = True
        trade_manager.is_simulated = False
        
        logger.info("Trading session started")
        logger.info(f"Watching symbols: {', '.join(args.symbols)}")
        
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
                
                # Scan for opportunities (focusing on provided symbols)
                logger.info(f"Scanning for trading opportunities in {', '.join(args.symbols)}...")
                
                # Get data for symbols
                stocks = []
                for symbol in args.symbols:
                    stock_data = market_data.get_stock_data(symbol)
                    if stock_data:
                        stocks.append(stock_data)
                
                if not stocks:
                    logger.warning(f"Failed to get data for {', '.join(args.symbols)}. Retrying in 60 seconds.")
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
                time.sleep(60)  # 1 minute between scans
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        
        # Stop trading session
        trade_manager.stop_trading_session()
        
        # Disconnect from IB
        ib_connector.disconnect()
        
        logger.info("Trading session ended")
        
    except Exception as e:
        logger.error(f"Error in live trading: {e}", exc_info=True)


if __name__ == '__main__':
    args = parse_arguments()
    run_live_trading(args)