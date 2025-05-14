#!/usr/bin/env python
"""
Stock Condition Scanner

This script scans stocks and checks if they meet Ross Cameron's 5 trading conditions:
1. Price between $2-$20
2. Gap up at least 10%
3. Relative volume at least 5x
4. Has breaking news
5. Float under 10 million shares

Results are saved to a CSV/JSON file for easy viewing.
Config.json is also automatically updated with stocks meeting all conditions.
"""

import os
import json
import csv
import time
import logging
from datetime import datetime
import pandas as pd
import requests

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockConditionScanner:
    def __init__(self, config_file="config.json"):
        """Initialize the scanner with configuration."""
        self.load_config(config_file)
        self.setup_data_providers()
        self.results = {}
        
    def load_config(self, config_file):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Set configuration parameters
            self.watchlist = config.get('watchlist', [])
            self.min_price = config.get('min_price', 2.0)
            self.max_price = config.get('max_price', 20.0)
            self.min_gap_pct = config.get('min_gap_pct', 10.0)
            self.min_rel_volume = config.get('min_rel_volume', 5.0)
            self.max_float = config.get('max_float', 10_000_000)
            self.output_format = config.get('output_format', 'json')
            self.output_file = config.get('output_file', 'stock_conditions.json')
            self.api_keys = config.get('api_keys', {})
            self.config_file = config_file
            
            logger.info(f"Loaded configuration with {len(self.watchlist)} stocks in watchlist")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Set defaults
            self.watchlist = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
            self.min_price = 2.0
            self.max_price = 20.0
            self.min_gap_pct = 10.0
            self.min_rel_volume = 5.0
            self.max_float = 10_000_000
            self.output_format = 'json'
            self.output_file = 'stock_conditions.json'
            self.api_keys = {}
            self.config_file = config_file
    
    def setup_data_providers(self):
        """Setup data providers for stock information."""
        # This would normally integrate with your data sources
        # For now, we'll use placeholder methods
        pass
    
    def check_price_condition(self, symbol):
        """Check if stock meets price condition ($2-$20)."""
        try:
            # Get current price (placeholder - implement with your data source)
            price = self.get_current_price(symbol)
            
            # Check condition
            condition_met = self.min_price <= price <= self.max_price
            
            return {
                "met": condition_met,
                "value": f"${price:.2f}",
                "requirement": f"Between ${self.min_price:.2f}-${self.max_price:.2f}"
            }
        except Exception as e:
            logger.error(f"Error checking price condition for {symbol}: {e}")
            return {"met": False, "value": "N/A", "requirement": f"Between ${self.min_price:.2f}-${self.max_price:.2f}"}
    
    def check_gap_condition(self, symbol):
        """Check if stock meets gap condition (up at least 10%)."""
        try:
            # Get gap percentage (placeholder)
            gap_pct = self.get_gap_percentage(symbol)
            
            # Check condition
            condition_met = gap_pct >= self.min_gap_pct
            
            return {
                "met": condition_met,
                "value": f"{gap_pct:.1f}%",
                "requirement": f"Up at least {self.min_gap_pct:.1f}%"
            }
        except Exception as e:
            logger.error(f"Error checking gap condition for {symbol}: {e}")
            return {"met": False, "value": "N/A", "requirement": f"Up at least {self.min_gap_pct:.1f}%"}
    
    def check_volume_condition(self, symbol):
        """Check if stock meets volume condition (rel. volume > 5x)."""
        try:
            # Get relative volume (placeholder)
            rel_volume = self.get_relative_volume(symbol)
            
            # Check condition
            condition_met = rel_volume >= self.min_rel_volume
            
            return {
                "met": condition_met,
                "value": f"{rel_volume:.1f}x",
                "requirement": f"Relative volume > {self.min_rel_volume:.1f}x"
            }
        except Exception as e:
            logger.error(f"Error checking volume condition for {symbol}: {e}")
            return {"met": False, "value": "N/A", "requirement": f"Relative volume > {self.min_rel_volume:.1f}x"}
    
    def check_news_condition(self, symbol):
        """Check if stock has breaking news."""
        try:
            # Check for news (placeholder)
            has_news = self.has_breaking_news(symbol)
            
            return {
                "met": has_news,
                "value": "Yes" if has_news else "No",
                "requirement": "Has breaking news"
            }
        except Exception as e:
            logger.error(f"Error checking news condition for {symbol}: {e}")
            return {"met": False, "value": "No", "requirement": "Has breaking news"}
    
    def check_float_condition(self, symbol):
        """Check if stock meets float condition (< 10M shares)."""
        try:
            # Get float size (placeholder)
            float_size = self.get_float_size(symbol)
            
            # Convert to millions for display
            float_millions = float_size / 1_000_000
            
            # Check condition
            condition_met = float_size <= self.max_float
            
            return {
                "met": condition_met,
                "value": f"{float_millions:.1f}M",
                "requirement": f"Float < {self.max_float/1_000_000:.1f}M shares"
            }
        except Exception as e:
            logger.error(f"Error checking float condition for {symbol}: {e}")
            return {"met": False, "value": "N/A", "requirement": f"Float < {self.max_float/1_000_000:.1f}M shares"}
    
    def scan_stock(self, symbol):
        """Scan a single stock for all conditions."""
        logger.info(f"Scanning {symbol}...")
        
        # Get current price for reference
        current_price = self.get_current_price(symbol)
        
        # Check all conditions
        price_condition = self.check_price_condition(symbol)
        gap_condition = self.check_gap_condition(symbol)
        volume_condition = self.check_volume_condition(symbol)
        news_condition = self.check_news_condition(symbol)
        float_condition = self.check_float_condition(symbol)
        
        # Check if all conditions are met
        all_conditions_met = (
            price_condition["met"] and
            gap_condition["met"] and
            volume_condition["met"] and
            news_condition["met"] and
            float_condition["met"]
        )
        
        # Create stock result
        stock_result = {
            "symbol": symbol,
            "conditions": {
                "price": price_condition,
                "percent_up": gap_condition,
                "volume": volume_condition,
                "news": news_condition,
                "float": float_condition
            },
            "all_conditions_met": all_conditions_met,
            "current_price": f"${current_price:.2f}",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes": ""
        }
        
        # Add to results
        self.results[symbol] = stock_result
        
        # Log result
        if all_conditions_met:
            logger.info(f"{symbol} meets ALL conditions!")
        else:
            # List which conditions failed
            failed_conditions = []
            for name, condition in stock_result["conditions"].items():
                if not condition["met"]:
                    failed_conditions.append(name)
                    
            logger.info(f"{symbol} failed conditions: {', '.join(failed_conditions)}")
        
        return stock_result
    
    def scan_watchlist(self):
        """Scan all stocks in the watchlist."""
        logger.info(f"Starting scan of {len(self.watchlist)} stocks...")
        
        for symbol in self.watchlist:
            try:
                self.scan_stock(symbol)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        logger.info("Scan complete")
        
        # Count stocks meeting all conditions
        stocks_meeting_all = sum(1 for stock in self.results.values() if stock["all_conditions_met"])
        logger.info(f"Found {stocks_meeting_all} stocks meeting all conditions")
        
        return self.results
    
    def save_results(self):
        """Save results to file in the specified format."""
        if self.output_format == 'json':
            self._save_json()
        elif self.output_format == 'csv':
            self._save_csv()
        else:
            logger.error(f"Unsupported output format: {self.output_format}")
    
    def _save_json(self):
        """Save results as JSON."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump({"stocks": list(self.results.values()), 
                          "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 
                         f, indent=2)
            logger.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
    
    def _save_csv(self):
        """Save results as CSV."""
        try:
            # Prepare CSV data
            csv_data = []
            
            for symbol, data in self.results.items():
                row = {
                    "Symbol": symbol,
                    "Price_Condition": "Yes" if data["conditions"]["price"]["met"] else "No",
                    "Price_Value": data["conditions"]["price"]["value"],
                    "Gap_Condition": "Yes" if data["conditions"]["percent_up"]["met"] else "No",
                    "Gap_Value": data["conditions"]["percent_up"]["value"],
                    "Volume_Condition": "Yes" if data["conditions"]["volume"]["met"] else "No",
                    "Volume_Value": data["conditions"]["volume"]["value"],
                    "News_Condition": "Yes" if data["conditions"]["news"]["met"] else "No",
                    "Float_Condition": "Yes" if data["conditions"]["float"]["met"] else "No",
                    "Float_Value": data["conditions"]["float"]["value"],
                    "All_Conditions_Met": "Yes" if data["all_conditions_met"] else "No",
                    "Current_Price": data["current_price"],
                    "Last_Updated": data["last_updated"],
                    "Notes": data["notes"]
                }
                csv_data.append(row)
            
            # Write to CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(self.output_file, index=False)
                logger.info(f"Results saved to {self.output_file}")
            else:
                logger.warning("No data to save to CSV")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
    
    def update_config_with_qualified_stocks(self):
        """
        Update config.json with qualified stocks that meet all 5 conditions.
        """
        try:
            # Get qualified stocks (those meeting all 5 conditions)
            qualified_stocks = [
                symbol for symbol, data in self.results.items() 
                if data["all_conditions_met"]
            ]
            
            if not qualified_stocks:
                logger.info("No qualified stocks found to update config")
                return
            
            # Read existing config
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            
            # Update watchlist with qualified stocks
            config['watchlist'] = qualified_stocks
            
            # Preserve other settings or set defaults
            config['min_price'] = config.get('min_price', self.min_price)
            config['max_price'] = config.get('max_price', self.max_price)
            config['min_gap_pct'] = config.get('min_gap_pct', self.min_gap_pct)
            config['min_rel_volume'] = config.get('min_rel_volume', self.min_rel_volume)
            config['max_float'] = config.get('max_float', self.max_float)
            config['output_format'] = config.get('output_format', self.output_format)
            config['output_file'] = config.get('output_file', self.output_file)
            config['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write updated config back to file
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Updated {self.config_file} with {len(qualified_stocks)} qualified stocks: {', '.join(qualified_stocks)}")
            
        except Exception as e:
            logger.error(f"Error updating config with qualified stocks: {e}")
    
    # Placeholder data provider methods (replace with your data source)
    def get_current_price(self, symbol):
        """Get current price for a stock (placeholder)."""
        # In a real implementation, this would call your data source
        import random
        return random.uniform(1.0, 25.0)
    
    def get_gap_percentage(self, symbol):
        """Get gap percentage for a stock (placeholder)."""
        import random
        return random.uniform(5.0, 15.0)
    
    def get_relative_volume(self, symbol):
        """Get relative volume for a stock (placeholder)."""
        import random
        return random.uniform(2.0, 8.0)
    
    def has_breaking_news(self, symbol):
        """Check if stock has breaking news (placeholder)."""
        import random
        return random.choice([True, False])
    
    def get_float_size(self, symbol):
        """Get float size for a stock (placeholder)."""
        import random
        return random.uniform(2_000_000, 15_000_000)


def main():
    """Main entry point."""
    # Create scanner instance
    scanner = StockConditionScanner()
    
    # If watchlist is empty, use a default broader list
    if not scanner.watchlist:
        scanner.watchlist = [
            "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META", 
            "NFLX", "BABA", "SHOP", "ROKU", "PLTR", "NIO", "LCID", "RIVN", 
            "COIN", "GME", "AMC", "BB", "NOK", "ZM", "SNAP", "PINS"
        ]
    
    # Run the scan
    scanner.scan_watchlist()
    
    # Save the scan results
    scanner.save_results()
    
    # Update config.json with qualified stocks
    scanner.update_config_with_qualified_stocks()


if __name__ == "__main__":
    main()