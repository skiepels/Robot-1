# run_backtest.py
import sys
from backtest.backtest_engine import BacktestEngine

def main():
    # Create an instance of BacktestEngine and run it
    from backtest.backtest_engine import main as backtest_main
    backtest_main()

if __name__ == "__main__":
    main()