# run_ross_strategy.py
import os
import sys
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.trading.trading_bot import TradingBot
from src.data.ib_connector import IBConnector

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Ross Cameron trading strategy')
    
    parser.add_argument('--symbol', type=str, default='GPUS',
                      help='Stock symbol to trade')
    
    parser.add_argument('--days', type=int, default=6,
                      help='Number of days to backtest')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital')
    
    return parser.parse_args()

def main():
    """Main function to run Ross Cameron's trading strategy."""
    args = parse_arguments()
    
    # Set up dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Format dates for display
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Running Ross Cameron's trading strategy on {args.symbol}")
    print(f"Period: {start_date_str} to {end_date_str}")
    print(f"Initial capital: ${args.capital:.2f}")
    
    # Initialize the trading bot
    bot = TradingBot(initial_capital=args.capital)
    
    # Initialize IB connector
    ib_connector = IBConnector(
        host=os.getenv('IB_HOST', '127.0.0.1'),
        port=int(os.getenv('IB_PORT', 7497)),
        client_id=int(os.getenv('IB_CLIENT_ID', 1))
    )
    
# Connect to Interactive Brokers
    if not ib_connector.connect():
        print("Failed to connect to Interactive Brokers. Exiting.")
        return
    
    try:
        # Run backtest
        print("Running backtest...")
        results = bot.backtest_strategy(
            ib_connector=ib_connector,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Display results
        if results:
            print("\n=== Backtest Results ===")
            print(f"Initial Capital: ${results['initial_capital']:.2f}")
            print(f"Final Capital: ${results['final_capital']:.2f}")
            print(f"Total P&L: ${results['total_pnl']:.2f}")
            print(f"Return: {results['return_pct']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.2f}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            
            # Display pattern-specific statistics
            if 'pattern_stats' in results:
                print("\n=== Pattern Performance ===")
                
                for pattern, stats in results['pattern_stats'].items():
                    print(f"\n{pattern.replace('_', ' ').title()} Pattern:")
                    print(f"  Trades: {stats['count']}")
                    print(f"  Win Rate: {stats['win_rate']:.2f}")
                    print(f"  Net P&L: ${stats['net_pnl']:.2f}")
                    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        else:
            print("No backtest results generated.")
            
    finally:
        # Disconnect from IB
        ib_connector.disconnect()
        print("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    main()