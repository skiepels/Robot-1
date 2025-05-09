#!/usr/bin/env python
"""
Run Pattern Backtest

This script runs a backtest using the specialized candlestick patterns:
Bull Flag, Bull Pennant, and Flat Top Breakout.

Usage:
python run_pattern_backtest.py --symbol GPUS --start-date 2023-05-02 --end-date 2023-05-08
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pattern_backtest import PatternBacktester

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run pattern-based backtest')
    
    parser.add_argument('--symbol', type=str, default='GPUS',
                      help='Stock symbol to backtest')
    
    parser.add_argument('--start-date', type=str, default='2023-05-02',
                      help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='2023-05-08',
                      help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital for backtest')
    
    return parser.parse_args()

def run_backtest():
    """Run the pattern backtest with the provided parameters."""
    # Parse command line arguments
    args = parse_args()
    
    # Extract parameters
    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    initial_capital = args.capital
    
    print(f"Running pattern backtest for {symbol} from {start_date} to {end_date}")
    
    # Create backtest engine
    backtest = PatternBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=[symbol]
    )
    
    # Run backtest
    results = backtest.run_backtest()
    
    return results

if __name__ == "__main__":
    run_backtest()