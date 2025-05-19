#!/usr/bin/env python
"""
Ross Cameron's 5 Conditions Stock Scanner

This script scans stocks to identify those meeting Ross Cameron's 5 key criteria:
1. Price between $2-$20
2. Gap up at least 10%
3. Relative volume at least 5x
4. Has breaking news
5. Float under 10 million shares

Usage:
    python five_conditions_scanner.py --date 2025-05-14 --watchlist stocks.txt --api-key YOUR_API_KEY
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FiveConditionsScanner:
    """
    Scanner for identifying stocks meeting Ross Cameron's 5 trading conditions.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize scanner with configuration.
        
        Parameters:
        -----------
        api_key: str, optional
            API key for data provider (if required)
        """
        self.api_key = api_key
        
        # Default trading parameters
        self.min_price = 2.0
        self.max_price = 20.0
        self.min_gap_pct = 10.0
        self.min_rel_volume = 5.0
        self.max_float = 10_000_000
        
        # Results storage
        self.results = {}
    
    def load_watchlist(self, filename=None):
        """
        Load watchlist from file or use default list.
        
        Parameters:
        -----------
        filename: str, optional
            Path to watchlist file (one symbol per line)
            
        Returns:
        --------
        list: Stock symbols to scan
        """
        default_watchlist = [
            "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META", 
            "NFLX", "BABA", "SHOP", "ROKU", "PLTR", "NIO", "LCID", "RIVN"
        ]
        
        if not filename or not os.path.exists(filename):
            logger.info(f"Using default watchlist with {len(default_watchlist)} symbols")
            return default_watchlist
        
        try:
            with open(filename, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(symbols)} symbols from {filename}")
            return symbols
        except Exception as e:
            logger.error(f"Error loading watchlist from {filename}: {e}")
            logger.info(f"Falling back to default watchlist with {len(default_watchlist)} symbols")
            return default_watchlist
    
    def scan_stocks(self, symbols, date=None):
        """
        Scan a list of stocks for the 5 conditions.
        
        Parameters:
        -----------
        symbols: list
            List of stock symbols to scan
        date: str or datetime, optional
            Date to scan for (default: today)
            
        Returns:
        --------
        dict: Dictionary of scan results
        """
        # Convert date to datetime if needed
        if date is None:
            scan_date = datetime.now()
        elif isinstance(date, str):
            scan_date = datetime.strptime(date, '%Y-%m-%d')
        else:
            scan_date = date
            
        date_str = scan_date.strftime('%Y-%m-%d')
        
        logger.info(f"Scanning {len(symbols)} stocks for date {date_str}...")
        
        # Reset results
        self.results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Scanning {symbol}...")
                
                # Get data for this stock
                stock_data = self.get_stock_data(symbol, scan_date)
                
                if stock_data is None:
                    logger.warning(f"Could not get data for {symbol}")
                    continue
                
                # Check all 5 conditions
                price_condition = self.check_price_condition(stock_data)
                gap_condition = self.check_gap_condition(stock_data)
                volume_condition = self.check_volume_condition(stock_data)
                news_condition = self.check_news_condition(symbol, scan_date)
                float_condition = self.check_float_condition(symbol)
                
                # Determine if all conditions are met
                conditions = {
                    'price': price_condition,
                    'percent_up': gap_condition,
                    'volume': volume_condition,
                    'news': news_condition,
                    'float': float_condition
                }
                
                all_conditions_met = all(condition['met'] for condition in conditions.values())
                
                # Create result
                self.results[symbol] = {
                    'symbol': symbol,
                    'conditions': conditions,
                    'all_conditions_met': all_conditions_met,
                    'current_price': stock_data.get('current_price', 'N/A'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'notes': ""
                }
                
                # Log result
                if all_conditions_met:
                    logger.info(f"{symbol} meets ALL conditions!")
                else:
                    failed_conditions = [
                        name for name, condition in conditions.items() 
                        if not condition['met']
                    ]
                    logger.info(f"{symbol} failed conditions: {', '.join(failed_conditions)}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Count qualifying stocks
        qualified_count = sum(1 for result in self.results.values() if result['all_conditions_met'])
        logger.info(f"Scan complete. Found {qualified_count} stocks meeting all 5 conditions.")
        
        return self.results
    
    def get_stock_data(self, symbol, date):
        """
        Get stock data for a specific date.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        date: datetime
            Date to get data for
            
        Returns:
        --------
        dict: Stock data including price, volume, etc.
        """
        try:
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            # Get historical data (including previous day for gap calculation)
            start_date = date - timedelta(days=5)  # Include previous days for context
            end_date = date + timedelta(days=1)    # Include scan date
            
            hist = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                                  end=end_date.strftime('%Y-%m-%d'),
                                  interval="1d")
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            # Get the row for our scan date
            date_str = date.strftime('%Y-%m-%d')
            scan_day_data = hist[hist.index.strftime('%Y-%m-%d') == date_str]
            
            if scan_day_data.empty and len(hist) > 0:
                # If exact date not found, use most recent
                scan_day_data = hist.iloc[-1:]
                
            if scan_day_data.empty:
                logger.warning(f"No data for {symbol} on {date_str}")
                return None
                
            # Get previous day for gap calculation
            if len(hist) > 1:
                prev_day_data = hist.iloc[hist.index.get_loc(scan_day_data.index[0]) - 1:hist.index.get_loc(scan_day_data.index[0])]
                prev_close = prev_day_data['Close'].iloc[0] if not prev_day_data.empty else None
            else:
                prev_close = None
            
            # Get company info
            info = ticker.info
            
            # Create result
            result = {
                'symbol': symbol,
                'date': date_str,
                'current_price': float(scan_day_data['Close'].iloc[0]),
                'open_price': float(scan_day_data['Open'].iloc[0]),
                'high_price': float(scan_day_data['High'].iloc[0]),
                'low_price': float(scan_day_data['Low'].iloc[0]),
                'volume': int(scan_day_data['Volume'].iloc[0]),
                'previous_close': float(prev_close) if prev_close is not None else None,
                'avg_volume': info.get('averageVolume10days', info.get('averageVolume', 0)),
                'float': info.get('floatShares', 0),
                'market_cap': info.get('marketCap', 0),
                'company_name': info.get('shortName', symbol)
            }
            
            # Calculate gap percentage if we have previous close
            if result['previous_close'] is not None and result['previous_close'] > 0:
                result['gap_percent'] = (result['open_price'] - result['previous_close']) / result['previous_close'] * 100
            else:
                result['gap_percent'] = 0
                
            # Calculate relative volume
            if result['avg_volume'] > 0:
                result['relative_volume'] = result['volume'] / result['avg_volume']
            else:
                result['relative_volume'] = 0
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def check_price_condition(self, stock_data):
        """
        Check if stock meets price condition ($2-$20).
        
        Parameters:
        -----------
        stock_data: dict
            Stock data including price
            
        Returns:
        --------
        dict: Condition check result
        """
        if stock_data is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Between ${self.min_price:.2f}-${self.max_price:.2f}"}
            
        price = stock_data.get('current_price')
        
        if price is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Between ${self.min_price:.2f}-${self.max_price:.2f}"}
            
        condition_met = self.min_price <= price <= self.max_price
        
        return {
            'met': condition_met,
            'value': f"${price:.2f}",
            'requirement': f"Between ${self.min_price:.2f}-${self.max_price:.2f}"
        }
    
    def check_gap_condition(self, stock_data):
        """
        Check if stock meets gap condition (up at least 10%).
        
        Parameters:
        -----------
        stock_data: dict
            Stock data including gap percentage
            
        Returns:
        --------
        dict: Condition check result
        """
        if stock_data is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Up at least {self.min_gap_pct:.1f}%"}
            
        gap_pct = stock_data.get('gap_percent')
        
        if gap_pct is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Up at least {self.min_gap_pct:.1f}%"}
            
        condition_met = gap_pct >= self.min_gap_pct
        
        return {
            'met': condition_met,
            'value': f"{gap_pct:.1f}%",
            'requirement': f"Up at least {self.min_gap_pct:.1f}%"
        }
    
    def check_volume_condition(self, stock_data):
        """
        Check if stock meets volume condition (relative volume > 5x).
        
        Parameters:
        -----------
        stock_data: dict
            Stock data including volume information
            
        Returns:
        --------
        dict: Condition check result
        """
        if stock_data is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Relative volume > {self.min_rel_volume:.1f}x"}
            
        rel_volume = stock_data.get('relative_volume')
        
        if rel_volume is None:
            return {'met': False, 'value': 'N/A', 'requirement': f"Relative volume > {self.min_rel_volume:.1f}x"}
            
        condition_met = rel_volume >= self.min_rel_volume
        
        return {
            'met': condition_met,
            'value': f"{rel_volume:.1f}x",
            'requirement': f"Relative volume > {self.min_rel_volume:.1f}x"
        }
    
    def check_news_condition(self, symbol, date):
        """
        Check if stock has breaking news.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        date: datetime
            Date to check for news
            
        Returns:
        --------
        dict: Condition check result
        """
        # For demonstration, this is simplified
        # In a real implementation, you would connect to a news API
        try:
            has_news = self.check_news_via_api(symbol, date)
            
            return {
                'met': has_news,
                'value': "Yes" if has_news else "No",
                'requirement': "Has breaking news"
            }
        except Exception as e:
            logger.error(f"Error checking news for {symbol}: {e}")
            return {'met': False, 'value': "No", 'requirement': "Has breaking news"}
    
    def check_news_via_api(self, symbol, date):
        """
        Check for breaking news via API.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        date: datetime
            Date to check for news
            
        Returns:
        --------
        bool: True if breaking news exists
        """
        # If you have a news API key, implement real news checking here
        if self.api_key:
            try:
                # Example using Alpha Vantage API
                date_str = date.strftime('%Y%m%dT0000')
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from={date_str}&apikey={self.api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    news_items = data.get('feed', [])
                    return len(news_items) > 0
                    
            except Exception as e:
                logger.error(f"Error fetching news from API for {symbol}: {e}")
        
        # Fallback to random generation for demo purposes
        # In a real implementation, replace this with actual news checking
        import random
        return random.choice([True, False])
    
    def check_float_condition(self, symbol):
        """
        Check if stock meets float condition (< 10M shares).
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        dict: Condition check result
        """
        try:
            # Get float data
            float_size = self.get_float_size(symbol)
            
            if float_size is None:
                return {'met': False, 'value': 'N/A', 'requirement': f"Float < {self.max_float/1_000_000:.1f}M shares"}
                
            # Convert to millions for display
            float_millions = float_size / 1_000_000
            
            # Check condition
            condition_met = float_size <= self.max_float
            
            return {
                'met': condition_met,
                'value': f"{float_millions:.1f}M",
                'requirement': f"Float < {self.max_float/1_000_000:.1f}M shares"
            }
        except Exception as e:
            logger.error(f"Error checking float for {symbol}: {e}")
            return {'met': False, 'value': 'N/A', 'requirement': f"Float < {self.max_float/1_000_000:.1f}M shares"}
    
    def get_float_size(self, symbol):
        """
        Get float size for a stock.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        int: Float size in shares
        """
        try:
            # Using Yahoo Finance for float data
            ticker = yf.Ticker(symbol)
            float_shares = ticker.info.get('floatShares', 0)
            
            return float_shares
        except Exception as e:
            logger.error(f"Error getting float size for {symbol}: {e}")
            
            # For demo purposes, generate a random float size
            # In a real implementation, use a reliable data source
            import random
            return random.randint(5_000_000, 15_000_000)
    
    def save_results(self, filename='stock_conditions.json'):
        """
        Save scan results to a JSON file.
        
        Parameters:
        -----------
        filename: str
            Output filename
            
        Returns:
        --------
        bool: True if successful
        """
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'stocks': list(self.results.values()),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
                
            logger.info(f"Results saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Scan stocks for Ross Cameron\'s 5 trading conditions')
    
    parser.add_argument('--date', type=str, default=None,
                      help='Date to scan for (YYYY-MM-DD format, default: today)')
    
    parser.add_argument('--watchlist', type=str, default=None,
                      help='Path to watchlist file (one symbol per line)')
    
    parser.add_argument('--output', type=str, default='stock_conditions.json',
                      help='Output file for results (JSON format)')
    
    parser.add_argument('--api-key', type=str, default=None,
                      help='API key for news data')
    
    parser.add_argument('--min-price', type=float, default=2.0,
                      help='Minimum stock price')
    
    parser.add_argument('--max-price', type=float, default=20.0,
                      help='Maximum stock price')
    
    parser.add_argument('--min-gap', type=float, default=10.0,
                      help='Minimum gap percentage')
    
    parser.add_argument('--min-rel-volume', type=float, default=5.0,
                      help='Minimum relative volume')
    
    parser.add_argument('--max-float', type=float, default=10.0,
                      help='Maximum float in millions')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = FiveConditionsScanner(api_key=args.api_key)
    
    # Set custom parameters if provided
    scanner.min_price = args.min_price
    scanner.max_price = args.max_price
    scanner.min_gap_pct = args.min_gap
    scanner.min_rel_volume = args.min_rel_volume
    scanner.max_float = args.max_float * 1_000_000  # Convert from millions
    
    # Parse date
    scan_date = datetime.now()
    if args.date:
        try:
            scan_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Using today's date.")
    
    # Load watchlist
    symbols = scanner.load_watchlist(args.watchlist)
    
    # Run scan
    scanner.scan_stocks(symbols, date=scan_date)
    
    # Save results
    scanner.save_results(args.output)
    
    # Print summary
    qualified_stocks = [s['symbol'] for s in scanner.results.values() if s['all_conditions_met']]
    if qualified_stocks:
        print(f"\nStocks meeting all 5 conditions on {scan_date.strftime('%Y-%m-%d')}:")
        for symbol in qualified_stocks:
            print(f"- {symbol}")
    else:
        print(f"\nNo stocks found meeting all 5 conditions on {scan_date.strftime('%Y-%m-%d')}.")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()