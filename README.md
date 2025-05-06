# Momentum Day Trading Strategy

A Python implementation of a day trading strategy focused on momentum stocks and technical patterns.

## Overview

This project implements a day trading strategy based on Ross Cameron's approach, which focuses on:

1. Finding high-momentum stocks with key criteria:
   - Gap up of at least 10%
   - 5x relative volume
   - Price between $1-$20
   - Low float (under 10 million shares)
   - News catalyst

2. Identifying specific chart patterns:
   - Bull flag: A consolidation pattern after a strong upward move
   - First candle to make new high: A breakout candle making a new high after a pullback
   - Micro pullback: A small retracement in an uptrend, followed by continuation
   - Hammer: A bullish reversal candle with a long lower wick
   - Shooting star: A bearish reversal candle with a long upper wick
   - Doji: A candle with little or no body, indicating indecision
   - Engulfing patterns: Bullish or bearish candles that completely engulf the previous candle
   - Tweezer patterns: Pairs of candles with similar highs or lows, indicating potential reversals

3. Managing risk with:
   - 2:1 profit-to-loss ratio
   - Small position sizing until profit "cushion" is established
   - Stop losses at specific technical levels
   - Daily loss limits
   - Maximum 3 consecutive losses rule

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance (or alternative market data source)
- requests
- pyyaml
- matplotlib

## Installation

1. Clone the repository: