"""
Interactive Brokers API Interface

This module provides an interface to Interactive Brokers for executing trades
as part of the trading strategy.
"""

import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from ib_insync import IB, Order, Contract, Stock, LimitOrder, MarketOrder, StopOrder, StopLimitOrder

# Import the IBConnector
from src.data.ib_connector import IBConnector

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class IBBroker:
    """Interface to Interactive Brokers API for order execution."""
    
    def __init__(self, connector=None, host=None, port=None, client_id=None):
        """
        Initialize the IB Broker interface.
        
        Parameters:
        -----------
        connector: IBConnector, optional
            Existing connector instance (if None, creates new one)
        host: str, optional
            Host address for TWS/IB Gateway (default: from .env)
        port: int, optional
            Port for TWS/IB Gateway (default: from .env)
        client_id: int, optional
            Client ID for IB API connection (default: from .env)
        """
        # Use provided connector or create new one
        if connector and isinstance(connector, IBConnector):
            self.connector = connector
        else:
            self.connector = IBConnector(host=host, port=port, client_id=client_id)
        
        # Ensure we have a connection
        self.connected = self.connector.connect()
        
        # Get IB instance from connector
        self.ib = self.connector.ib
        
        # Track orders and positions
        self.orders = {}
        self.positions = {}
        
        # Initialize account data
        self.account_summary = {}
        
        logger.info("IBBroker initialized")
    
    def connect(self):
        """
        Connect to Interactive Brokers.
        
        Returns:
        --------
        bool: Connection status
        """
        if self.connected:
            return True
            
        self.connected = self.connector.connect()
        
        if self.connected:
            # Request account updates when connected
            self.request_account_updates()
        
        return self.connected
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        self.connector.disconnect()
        self.connected = False
    
    def request_account_updates(self):
        """Request account updates from IB."""
        if not self.ensure_connection():
            return False
            
        try:
            # Request account summary
            self.ib.reqAccountSummary()
            
            # Request positions
            self.ib.reqPositions()
            
            return True
        except Exception as e:
            logger.error(f"Error requesting account updates: {str(e)}")
            return False
    
    def get_account_balance(self):
        """
        Get the current account balance.
        
        Returns:
        --------
        float: Current cash balance
        """
        if not self.ensure_connection():
            return 0.0
            
        try:
            # Get account summary
            account_values = self.ib.accountSummary()
            
            # Find TotalCashValue tag
            for value in account_values:
                if value.tag == 'TotalCashValue':
                    return float(value.value)
            
            logger.warning("Could not find TotalCashValue in account summary")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0.0
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
        --------
        dict: Dictionary of positions by symbol
        """
        if not self.ensure_connection():
            return {}
            
        try:
            # Get positions
            positions = self.ib.positions()
            
            # Convert to dictionary
            positions_dict = {}
            for pos in positions:
                symbol = pos.contract.symbol
                positions_dict[symbol] = {
                    'symbol': symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue if hasattr(pos, 'marketValue') else 0.0,
                    'unrealized_pnl': pos.unrealizedPNL if hasattr(pos, 'unrealizedPNL') else 0.0,
                    'contract': pos.contract
                }
            
            # Update internal positions
            self.positions = positions_dict
            
            return positions_dict
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def submit_order(self, symbol, quantity, side, order_type, time_in_force='DAY', 
                    limit_price=None, stop_price=None, outside_rth=False):
        """
        Submit an order to Interactive Brokers.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        quantity: int
            Number of shares to trade
        side: str
            'BUY' or 'SELL'
        order_type: str
            'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
        time_in_force: str
            'DAY', 'GTC', 'IOC', etc.
        limit_price: float, optional
            Price for limit orders
        stop_price: float, optional
            Price for stop orders
        outside_rth: bool
            Allow trading outside regular trading hours
            
        Returns:
        --------
        dict: Order details
        """
        if not self.ensure_connection():
            return None
            
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order
            if order_type.upper() == 'MARKET':
                order = MarketOrder(side, quantity)
            elif order_type.upper() == 'LIMIT' and limit_price is not None:
                order = LimitOrder(side, quantity, limit_price)
            elif order_type.upper() == 'STOP' and stop_price is not None:
                order = StopOrder(side, quantity, stop_price)
            elif order_type.upper() == 'STOP_LIMIT' and limit_price is not None and stop_price is not None:
                order = StopLimitOrder(side, quantity, limit_price, stop_price)
            else:
                logger.error(f"Invalid order parameters: {order_type}")
                return None
            
            # Set time in force
            order.tif = time_in_force
            
            # Set outside RTH flag
            order.outsideRth = outside_rth
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be acknowledged
            timeout = 5  # seconds
            start_time = time.time()
            while not trade.orderStatus.status and time.time() - start_time < timeout:
                self.ib.sleep(0.1)
            
            # Create order info
            order_info = {
                'id': trade.order.orderId,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining,
                'avg_fill_price': trade.orderStatus.avgFillPrice,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to orders dictionary
            self.orders[trade.order.orderId] = order_info
            
            logger.info(f"Order submitted: {symbol} {side} {quantity} shares at {limit_price or stop_price or 'market'}")
            
            return order_info
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel an existing order.
        
        Parameters:
        -----------
        order_id: int
            Order ID to cancel
            
        Returns:
        --------
        bool: Success or failure
        """
        if not self.ensure_connection():
            return False
            
        try:
            # Find order
            open_trades = self.ib.openTrades()
            for trade in open_trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True
            
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, order_id):
        """
        Get the status of an order.
        
        Parameters:
        -----------
        order_id: int
            Order ID to check
            
        Returns:
        --------
        dict: Order status
        """
        if not self.ensure_connection():
            return None
            
        try:
            # Check our own records first
            if order_id in self.orders:
                return self.orders[order_id]
            
            # Check with IB
            open_trades = self.ib.openTrades()
            for trade in open_trades:
                if trade.order.orderId == order_id:
                    status = {
                        'id': order_id,
                        'status': trade.orderStatus.status,
                        'filled': trade.orderStatus.filled,
                        'remaining': trade.orderStatus.remaining,
                        'avg_fill_price': trade.orderStatus.avgFillPrice,
                        'last_fill_price': trade.orderStatus.lastFillPrice
                    }
                    
                    # Update our records
                    if order_id in self.orders:
                        self.orders[order_id].update(status)
                    
                    return status
            
            logger.warning(f"Order {order_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    def get_executions(self, client_id=None):
        """
        Get recent executions.
        
        Parameters:
        -----------
        client_id: int, optional
            Filter by client ID
            
        Returns:
        --------
        list: Execution details
        """
        if not self.ensure_connection():
            return []
            
        try:
            # Get executions
            executions = self.ib.executions()
            
            # Filter by client ID if provided
            if client_id is not None:
                executions = [ex for ex in executions if ex.execution.clientId == client_id]
            
            # Convert to list of dictionaries
            exec_list = []
            for exec_detail in executions:
                ex = exec_detail.execution
                exec_list.append({
                    'id': ex.execId,
                    'order_id': ex.orderId,
                    'symbol': exec_detail.contract.symbol,
                    'side': ex.side,
                    'shares': ex.shares,
                    'price': ex.price,
                    'time': ex.time.strftime('%Y-%m-%d %H:%M:%S'),
                    'commission': exec_detail.commissionReport.commission if hasattr(exec_detail, 'commissionReport') else 0.0
                })
            
            return exec_list
        except Exception as e:
            logger.error(f"Error getting executions: {str(e)}")
            return []
    
    def get_market_data(self, symbol):
        """
        Get current market data for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        dict: Market data
        """
        if not self.ensure_connection():
            return None
            
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            self.ib.reqMktData(contract)
            
            # Wait for data
            timeout = 3  # seconds
            start_time = time.time()
            ticker = self.ib.ticker(contract)
            
            while (ticker.bid <= 0 or ticker.ask <= 0) and time.time() - start_time < timeout:
                self.ib.sleep(0.1)
                ticker = self.ib.ticker(contract)
            
            # Cancel market data request
            self.ib.cancelMktData(contract)
            
            # Create market data dictionary
            market_data = {
                'symbol': symbol,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'close': ticker.close,
                'volume': ticker.volume,
                'bid_size': ticker.bidSize,
                'ask_size': ticker.askSize,
                'last_size': ticker.lastSize,
                'high': ticker.high,
                'low': ticker.low,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def ensure_connection(self):
        """Ensure connection to IB is active, reconnect if necessary."""
        if not self.connected or not self.ib.isConnected():
            logger.info("Connection to IB not active, attempting to reconnect")
            return self.connect()
        return True
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.disconnect()