# Momentum Day Trading Strategy

A Python implementation day trading strategy focused on momentum stocks and technical patterns.

## Overview

This project implements a day trading strategy based on Ross Cameron's approach, which focuses on:

1. Finding high-momentum stocks with key criteria:
   - Gap up of at least 10%
   - 5x relative volume
   - Price between $1-$20
   - Low float (under 10 million shares)
   - News catalyst

2. Identifying specific chart patterns:
   - Bull flag
   - First candle to make new high
   - Micro pullback

3. Managing risk with:
   - 2:1 profit-to-loss ratio
   - Small position sizing until profit "cushion" is established
   - Stop losses at specific technical levels
   - Daily loss limits
   - Maximum 3 consecutive losses rule

## Project Structure

```
trading_strategy/
├── main.py              # Main entry point
├── config.yaml          # Configuration settings
│
├── src/
│   ├── data/            # Data retrieval and processing
│   │   ├── market_data.py
│   │   ├── stock.py
│   │   └── news_data.py
│   │
│   ├── scanning/        # Market scanning and condition tracking
│   │   ├── scanner.py
│   │   └── condition_tracker.py
│   │
│   ├── analysis/        # Technical analysis and pattern recognition
│   │   └── candlestick_patterns.py
│   │
│   ├── trading/         # Trade execution and management
│   │   ├── trade_manager.py
│   │   └── risk_manager.py
│   │
│   └── utils/           # Utility functions
│       └── logger.py
│
└── tests/               # Unit tests
    ├── test_stock.py
    ├── test_condition_tracker.py
    ├── test_candlestick_patterns.py
    └── test_risk_manager.py
```

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance (or alternative market data source)
- requests
- pyyaml

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/momentum-day-trading.git
   cd momentum-day-trading
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your settings in `config.yaml`

## Usage

### Simulation Mode

Run the strategy in simulation mode (default):

```
python main.py
```

### Scan Only Mode

Scan for trading opportunities without executing trades:

```
python main.py --scan-only
```

### Custom Configuration

Use a custom configuration file:

```
python main.py --config my_config.yaml
```

### Live Trading

For live trading (requires broker API setup and configuration):

```
python main.py --simulate false
```

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk and may result in the loss of your capital. The developers of this software do not guarantee its accuracy or performance in live trading. Use at your own risk.

## License

MIT