#!/usr/bin/env python
"""
Run Dynamic Backtest

This script runs the dynamic backtester with user-specified parameters.

Usage:
python run_dynamic_backtest.py --symbol GPUS --start-date 2023-05-02 --end-date 2023-05-08 --strategy bull_flag
python run_dynamic_backtest.py --list-strategies
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import json

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the dynamic backtester
from dynamic_backtest import DynamicBacktester


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run dynamic backtesting')
    
    parser.add_argument('--symbol', type=str, default='GPUS',
                      help='Stock symbol to backtest')
    
    parser.add_argument('--symbols', type=str, nargs='+',
                      help='Multiple stock symbols to backtest')
    
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--days', type=int, default=5,
                      help='Number of trading days to backtest (if start/end dates not provided)')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                      help='Initial capital for backtest')
    
    parser.add_argument('--strategy', type=str, default=None,
                      help='Trading strategy to use')
    
    parser.add_argument('--list-strategies', action='store_true',
                      help='List all available strategies')
    
    parser.add_argument('--output', type=str, default=None,
                      help='Output file for results (JSON format)')
    
    return parser.parse_args()


def main():
    """Run the dynamic backtester with the provided parameters."""
    # Parse command line arguments
    args = parse_args()
    
    # Create backtester instance
    backtester = DynamicBacktester(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        symbols=args.symbols if args.symbols else [args.symbol],
        strategy=args.strategy
    )
    
    # If requested, list available strategies and exit
    if args.list_strategies:
        print("\nAvailable Trading Strategies:")
        strategies = backtester.list_available_strategies()
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
        print()
        return
    
    # Set dates if not provided
    if not args.start_date:
        backtester.start_date = datetime.now() - timedelta(days=args.days)
    
    if not args.end_date:
        backtester.end_date = datetime.now()
    
    # If no strategy provided, prompt for one
    if not args.strategy:
        available_strategies = backtester.list_available_strategies()
        
        if not available_strategies:
            print("No strategies found. Make sure you have strategy classes in the src/entry folder.")
            return
            
        print("\nAvailable Trading Strategies:")
        for i, strategy in enumerate(available_strategies, 1):
            print(f"{i}. {strategy}")
            
        choice = input("\nSelect a strategy (number or name): ")
        try:
            # Try to interpret as a number
            idx = int(choice) - 1
            if 0 <= idx < len(available_strategies):
                strategy = available_strategies[idx]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            # Interpret as a name
            strategy = choice.lower()
            if strategy not in available_strategies:
                print(f"Strategy '{strategy}' not found.")
                return
    else:
        strategy = args.strategy
    
    # Set the strategy
    if not backtester.set_strategy(strategy):
        print(f"Failed to set strategy: {strategy}")
        return
    
    # Run the backtest
    print(f"Running backtest for {', '.join(backtester.symbols)} with strategy '{backtester.strategy_name}'...")
    results = backtester.run_backtest()
    
    # Display results
    backtester.display_results(results)
    
    # Save results to file if requested
    if args.output and results:
        try:
            # Convert datetime objects to strings
            for trade in results['trades']:
                if isinstance(trade['entry_time'], datetime):
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if isinstance(trade['exit_time'], datetime):
                    trade['exit_time'] = trade['exit_time'].isoformat()
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()