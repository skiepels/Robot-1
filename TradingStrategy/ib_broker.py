"""
Interactive Brokers API Interface

This module provides an interface to Interactive Brokers for executing trades
as part of the Ross Cameron day trading strategy.
"""

import logging
import threading
import time
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData

logger = logging.getLogger(__name__)

class IBBroker(EWrapper, EClient):
    """Interface to Interactive Brokers API."""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize the IB Broker interface.
        
        Parameters:
        -----------
        host: str
            Host address for TWS/IB Gateway (default: 127.0.0.1)
        port: int
            Port for TWS/IB Gateway (default: 7497 for paper trading, 7496 for live)
        client_id: int
            Client ID for IB API connection
        """
        EClient.__init__(self, self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # Request ID counters
        self.next_order_id = None
        self.next_request_id = 1
        
        # Data storage
        self.positions = {}
        self.account_summary = {}
        self.market_data = {}
        self.order_status = {}
        self.executions = {}
        
        # Event flags
        self.data_ready = threading.Event()
        
        logger.info(f"IBBroker initialized with host={host}, port={port}, client_id={client_id}")
    
    def connect(self):
        """Connect to Interactive Brokers."""
        if self.connected:
            logger.info("Already connected to IB")
            return True
            
        logger.info(f"Connecting to IB at {self.host}:{self.port}")
        
        # Connect to TWS/IB Gateway
        self.connect(self.host, self.port, self.client_id)
        
        # Wait for nextValidId to confirm connection
        timeout = 5  # seconds
        start_time = time.time()
        
        while self.next_order_id is None and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if self.next_order_id is None:
            logger.error("Failed to connect to IB within timeout")
            return False
        
        self.connected = True
        logger.info(f"Connected to IB (next order ID: {self.next_order_id})")
        
        # Request account updates
        self.reqAccountUpdates(True, "")
        
        return True
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.connected:
            logger.info("Disconnecting from IB")
            self.reqAccountUpdates(False, "")
            self.disconnect()
            self.connected = False
    
    def create_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD'):
        """
        Create a contract object for the specified security.
        
        Parameters:
        -----------
        symbol: str
            Symbol of the security
        sec_type: str
            Security type (default: 'STK' for stocks)
        exchange: str
            Exchange (default: 'SMART' for smart routing)
        currency: str
            Currency (default: 'USD')
            
        Returns:
        --------
        Contract: IB contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        return contract
    
    def create_order(self, action, quantity, order_type, price=None, stop_price=None):
        """
        Create an order object.
        
        Parameters:
        -----------
        action: str
            Action ('BUY' or 'SELL')
        quantity: int
            Number of shares/contracts
        order_type: str
            Order type ('MKT', 'LMT', 'STP', 'STP_LMT', etc.)
        price: float, optional
            Limit price (for limit orders)
        stop_price: float, optional
            Stop price (for stop orders)
            
        Returns:
        --------
        Order: IB order object
        """
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        
        if price is not None:
            order.lmtPrice = price
            
        if stop_price is not None:
            order.auxPrice = stop_price
        
        # Add additional properties for day trading
        order.tif = 'DAY'  # Time in force: day only
        order.transmit = True
        
        return order
    
    def submit_order(self, symbol, quantity, side, order_type, time_in_force='DAY', limit_price=None, stop_price=None):
        """
        Submit an order to Interactive Brokers.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        quantity: int
            Number of shares
        side: str
            'buy' or 'sell'
        order_type: str
            'market', 'limit', 'stop', 'stop_limit'
        time_in_force: str
            Time in force (default: 'DAY')
        limit_price: float, optional
            Limit price (for limit orders)
        stop_price: float, optional
            Stop price (for stop orders)
            
        Returns:
        --------
        dict: Order information
        """
        if not self.connected:
            if not self.connect():
                logger.error("Cannot submit order: not connected to IB")
                return None
        
        # Map parameters to IB API format
        action = 'BUY' if side.lower() == 'buy' else 'SELL'
        
        # Map order type
        ib_order_type = {
            'market': 'MKT',
            'limit': 'LMT',
            'stop': 'STP',
            'stop_limit': 'STP_LMT'
        }.get(order_type.lower(), 'MKT')
        
        # Create contract
        contract = self.create_contract(symbol)
        
        # Create order
        order = self.create_order(
            action=action,
            quantity=quantity,
            order_type=ib_order_type,
            price=limit_price,
            stop_price=stop_price
        )
        
        # Submit order
        order_id = self.next_order_id
        self.next_order_id += 1
        
        logger.info(f"Submitting order: {action} {quantity} {symbol} {ib_order_type}")
        self.placeOrder(order_id, contract, order)
        
        # Create order record
        order_info = {
            'id': str(order_id),
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'time_in_force': time_in_force,
            'status': 'submitted',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store order status
        self.order_status[order_id] = order_info
        
        return order_info
    
    def get_market_data(self, symbol, data_type='REALTIME'):
        """
        Get market data for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        data_type: str
            Type of data ('REALTIME' or 'HISTORICAL')
            
        Returns:
        --------
        dict: Market data
        """
        if not self.connected:
            if not self.connect():
                logger.error("Cannot get market data: not connected to IB")
                return None
        
        # Create contract
        contract = self.create_contract(symbol)
        
        # Request ID for this data request
        req_id = self.next_request_id
        self.next_request_id += 1
        
        # Clear any previous data
        if symbol in self.market_data:
            del self.market_data[symbol]
        
        # Request market data
        self.data_ready.clear()
        self.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for data (with timeout)
        if not self.data_ready.wait(timeout=5):
            logger.warning(f"Timeout waiting for market data for {symbol}")
            return None
        
        return self.market_data.get(symbol, None)
    
    def get_account_balance(self):
        """
        Get account balance.
        
        Returns:
        --------
        float: Account balance
        """
        if not self.connected:
            if not self.connect():
                logger.error("Cannot get account balance: not connected to IB")
                return 0.0
        
        # Request account summary if not available
        if not self.account_summary:
            self.data_ready.clear()
            req_id = self.next_request_id
            self.next_request_id += 1
            
            self.reqAccountSummary(req_id, "All", "TotalCashValue")
            
            # Wait for data (with timeout)
            if not self.data_ready.wait(timeout=5):
                logger.warning("Timeout waiting for account summary")
                return 0.0
        
        # Return current cash balance
        return float(self.account_summary.get('TotalCashValue', 0.0))
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
        --------
        dict: Current positions
        """
        if not self.connected:
            if not self.connect():
                logger.error("Cannot get positions: not connected to IB")
                return {}
        
        # Request positions if not available
        if not self.positions:
            self.data_ready.clear()
            self.reqPositions()
            
            # Wait for data (with timeout)
            if not self.data_ready.wait(timeout=5):
                logger.warning("Timeout waiting for positions")
                return {}
        
        return self.positions
    
    #
    # IB API Callback Methods
    #
    
    def nextValidId(self, orderId):
        """Callback for the next valid order ID."""
        self.next_order_id = orderId
    
    def error(self, req_id, error_code, error_string):
        """Callback for errors."""
        logger.error(f"IB Error {error_code}: {error_string} (reqId: {req_id})")
    
    def accountSummary(self, req_id, account, tag, value, currency):
        """Callback for account summary data."""
        self.account_summary[tag] = value
        
        # Signal that we received data
        self.data_ready.set()
    
    def position(self, account, contract, position, avg_cost):
        """Callback for position data."""
        symbol = contract.symbol
        
        self.positions[symbol] = {
            'symbol': symbol,
            'position': position,
            'avg_cost': avg_cost
        }
        
        # Signal that we received position data
        self.data_ready.set()
    
    def tickPrice(self, req_id, tick_type, price, attrib):
        """Callback for price updates."""
        # Find contract for this request ID
        for symbol, data in self.market_data.items():
            if data.get('req_id') == req_id:
                # Update price data
                if tick_type == 1:  # Bid price
                    self.market_data[symbol]['bid'] = price
                elif tick_type == 2:  # Ask price
                    self.market_data[symbol]['ask'] = price
                elif tick_type == 4:  # Last price
                    self.market_data[symbol]['last'] = price
                elif tick_type == 9:  # Close price
                    self.market_data[symbol]['close'] = price
                
                # Signal that we received data
                self.data_ready.set()
                break
    
    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price, 
                    perm_id, parent_id, last_fill_price, client_id, why_held, mkt_cap_price):
        """Callback for order status updates."""
        if order_id in self.order_status:
            self.order_status[order_id].update({
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avg_fill_price': avg_fill_price,
                'last_fill_price': last_fill_price
            })
            
            logger.info(f"Order {order_id} status: {status}, filled: {filled}/{filled+remaining}")
    
    def execDetails(self, req_id, contract, execution):
        """Callback for execution details."""
        self.executions[execution.execId] = {
            'symbol': contract.symbol,
            'side': execution.side,
            'shares': execution.shares,
            'price': execution.price,
            'time': execution.time
        }
        
        logger.info(f"Execution: {execution.side} {execution.shares} {contract.symbol} @ {execution.price}")