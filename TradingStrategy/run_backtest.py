# run_backtest.py
import sys
import argparse
from backtest.backtest_engine import BacktestEngine

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategy')
    
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=None,
                      help='Initial capital for backtest')
    
    parser.add_argument('--symbols', type=str, nargs='+', default=['GPUS'],
                      help='Symbols to backtest, space separated')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create backtest engine
    engine = BacktestEngine(
        config_file=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        symbols=args.symbols
    )
    
    # Run backtest
    engine.run_backtest()

if __name__ == "__main__":
    main()