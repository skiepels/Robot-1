# run_trading_bot.py
import os
import sys
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import your existing backtest engine
from backtest.backtest_engine import BacktestEngine

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the trading bot for GPUS')
    
    parser.add_argument('--mode', type=str, choices=['live', 'backtest'], default='backtest',
                      help='Trading mode (live or backtest)')
    
    parser.add_argument('--symbol', type=str, default='GPUS',
                      help='Stock symbol to trade')
    
    parser.add_argument('--days', type=int, default=6,
                      help='Number of days to backtest')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital')
    
    parser.add_argument('--paper', action='store_true',
                      help='Use paper trading (for live mode)')
    
    return parser.parse_args()

def main():
    """Main function to run the trading bot."""
    args = parse_arguments()
    
    # Set up dates for backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Format dates for display
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Trading {args.symbol} from {start_date_str} to {end_date_str}")
    print(f"Initial capital: ${args.capital:.2f}")
    
    if args.mode == 'backtest':
        print("Running backtest...")
        
        # Create backtest engine with the parameters
        engine = BacktestEngine(
            config_file='config/config.yaml',
            start_date=start_date_str,
            end_date=end_date_str,
            initial_capital=args.capital,
            symbols=[args.symbol]
        )
        
        # Run the backtest
        results = engine.run_backtest()
        
        # Results are displayed by the backtest engine itself
        
    elif args.mode == 'live':
        print(f"Live trading mode is not yet implemented.")
        print(f"Please use run_live_trading.py instead with:")
        print(f"python run_live_trading.py --symbols {args.symbol} {'--paper' if args.paper else '--live'}")

if __name__ == "__main__":
    main()