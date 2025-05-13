# dashboard/app.py
from flask import Flask, render_template, jsonify, request
import threading
import time
import logging
import os
import sys

# Add the parent directory to path to import trading strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading strategy components
from src.data.ib_connector import IBConnector
from src.conditions.condition_checker import ConditionChecker
from src.scanning.scanner import StockScanner
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.trading.risk_manager import RiskManager
from src.utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__)
logger = setup_logger('dashboard', log_dir='logs')

# Global variables for data access
ib_connector = None
condition_checker = ConditionChecker()
market_data = None
news_data = None
scanner = None
risk_manager = None

# Store current stocks data
stocks_data = {}
qualified_stocks = {}
detected_patterns = {}
active_trades = {}

# Connection status
connection_status = {"connected": False, "message": "Not connected to Interactive Brokers"}

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/connect', methods=['POST'])
def connect_to_ib():
    """Connect to Interactive Brokers"""
    global ib_connector, market_data, news_data, scanner, connection_status
    
    try:
        # Initialize IB connector
        ib_connector = IBConnector()
        if ib_connector.connect():
            connection_status = {"connected": True, "message": "Connected to Interactive Brokers"}
            
            # Initialize other components
            market_data = MarketDataProvider(ib_connector=ib_connector)
            news_data = NewsDataProvider()
            scanner = StockScanner(market_data, news_data)
            risk_manager = RiskManager(initial_capital=10000.0)  # Default value
            
            # Start data monitoring thread
            threading.Thread(target=monitor_stocks, daemon=True).start()
            
            return jsonify({"success": True, "message": "Connected to Interactive Brokers"})
        else:
            connection_status = {"connected": False, "message": "Failed to connect to Interactive Brokers"}
            return jsonify({"success": False, "message": "Failed to connect to Interactive Brokers"})
    
    except Exception as e:
        logger.error(f"Error connecting to IB: {e}")
        connection_status = {"connected": False, "message": f"Error: {str(e)}"}
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/stocks', methods=['GET'])
def get_stocks():
    """Get current stocks data"""
    return jsonify({
        "connection": connection_status,
        "stocks": stocks_data,
        "qualified": qualified_stocks,
        "patterns": detected_patterns,
        "trades": active_trades
    })

@app.route('/add_stock', methods=['POST'])
def add_stock():
    """Add a stock to the watchlist"""
    stock = request.json.get('symbol')
    if stock:
        # Add to stocks to monitor
        stocks_data[stock] = {"symbol": stock, "conditions": {}, "last_updated": time.time()}
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid stock symbol"})

def monitor_stocks():
    """Background thread to monitor stocks and update conditions"""
    global stocks_data, qualified_stocks, detected_patterns
    
    while True:
        if ib_connector and ib_connector.connected:
            # Update each stock in our watchlist
            for symbol in list(stocks_data.keys()):
                try:
                    # Get stock data
                    stock_data = get_stock_data(symbol)
                    if stock_data:
                        # Check conditions
                        all_met, conditions = condition_checker.check_all_conditions(stock_data)
                        
                        # Update stock data
                        stocks_data[symbol] = {
                            "symbol": symbol,
                            "price": stock_data.get("current_price", 0),
                            "gap_percent": stock_data.get("day_change_percent", 0),
                            "rel_volume": stock_data.get("relative_volume", 0),
                            "has_news": stock_data.get("has_news", False),
                            "float": stock_data.get("shares_float", 0),
                            "conditions": conditions,
                            "all_conditions_met": all_met,
                            "last_updated": time.time()
                        }
                        
                        # If all conditions met, check for patterns
                        if all_met:
                            qualified_stocks[symbol] = stocks_data[symbol]
                            check_patterns(symbol)
                        elif symbol in qualified_stocks:
                            del qualified_stocks[symbol]
                            
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
        
        # Sleep for a short period before checking again
        time.sleep(5)

def get_stock_data(symbol):
    """Get stock data for condition checking"""
    try:
        # Replace with actual implementation using ib_connector
        # This is a placeholder implementation
        if not ib_connector or not ib_connector.connected:
            return None
            
        # Get current price
        current_price = ib_connector.get_current_price(symbol)
        if not current_price:
            return None
            
        # Get historical data for price change calculation
        hist_data = ib_connector.get_historical_data(
            symbol, duration='2 D', bar_size='1 day'
        )
        
        if hist_data.empty:
            return None
            
        # Calculate day change percentage
        previous_close = hist_data['close'].iloc[-2] if len(hist_data) >= 2 else 0
        day_change_percent = ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0
        
        # Get volume data
        current_volume = ib_connector.get_current_volume(symbol) or 0
        avg_volume = ib_connector.get_average_volume(symbol, period=50) or 0
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Get news (simplified)
        has_news = True  # Simplified for now
        
        # Get float size (from contract details)
        contract_details = ib_connector.get_contract_details(symbol)
        shares_float = contract_details.get('float', 0) if contract_details else 0
        
        return {
            "current_price": current_price,
            "day_change_percent": day_change_percent,
            "relative_volume": relative_volume,
            "has_news": has_news,
            "shares_float": shares_float
        }
    
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None

def check_patterns(symbol):
    """Check for trading patterns on a qualified stock"""
    try:
        # Get intraday data
        intraday_data = ib_connector.get_historical_data(
            symbol, duration='1 D', bar_size='1 min'
        )
        
        if intraday_data.empty:
            return
            
        # Use pattern detector to find patterns
        # This is simplified for now - in real implementation, use the PatternDetector
        patterns = [
            {
                "pattern": "Bull Flag",
                "confidence": 85,
                "entry_price": stocks_data[symbol]["price"] * 1.01,
                "stop_price": stocks_data[symbol]["price"] * 0.98,
                "target_price": stocks_data[symbol]["price"] * 1.05,
                "timestamp": time.time()
            }
        ]
        
        # Update detected patterns
        if patterns:
            detected_patterns[symbol] = patterns
        
    except Exception as e:
        logger.error(f"Error checking patterns for {symbol}: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)