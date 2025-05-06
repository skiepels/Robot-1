"""
Day Trading Strategy based on Ross Cameron's approach

Main entry point for the trading strategy implementation.
"""

import os
import sys
import time
import logging
import argparse
import yaml
from datetime import datetime, timedelta

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data.market_data import MarketDataProvider
from src.data.news_data import NewsDataProvider
from src.scanning.scanner import StockScanner
from src.scanning.condition_tracker import ConditionTracker
from src.trading.risk_manager import RiskManager
from src.trading.trade_manager import TradeManager
from src.utils.logger import setup_logger, get_trade_logger, get_performance_logger


def load_config(config_file='config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_file: str
        Path to configuration file
        
    Returns:
    --------
    dict: Configuration parameters
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        # Use default configuration
        return {
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs'
            },
            'market_data': {
                'api_key': None
            },
            'trading': {
                'initial_capital': 10000,
                'max_risk_per_trade_pct': 1.0,
                'daily_max_loss_pct': 3.0,
                'profit_loss_ratio': 2.0,
                'max_open_positions': 3,
                'min_price': 1.0,
                'max_price': 20.0,
                'min_gap_pct': 10.0,
                'min_rel_volume': 5.0,
                'max_float': 10000000
            },
            'simulation': {
                'enabled': True
            }
        }


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run Day Trading Strategy')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    
    parser.add_argument('--simulate', action='store_true',
                      help='Run in simulation mode')
    
    parser.add_argument('--capital', type=float,
                      help='Initial capital')
    
    parser.add_argument('--max-risk', type=float,
                      help='Maximum risk per trade percentage')
    
    parser.add_argument('--scan-only', action='store_true',
                      help='Scan for opportunities without trading')
    
    return parser.parse_args()


def run_strategy(config, args):
    """
    Run the day trading strategy based on configuration.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
    args: argparse.Namespace
        Command line arguments
    """
    # Set up logging
    log_level = getattr(logging, config['logging']['level'])
    log_dir = config['logging']['log_dir']
    
    logger = setup_logger('trading_strategy', log_dir=log_dir, console_level=log_level)
    trade_logger = get_trade_logger(log_dir=log_dir)
    performance_logger = get_performance_logger(log_dir=log_dir)
    
    logger.info("Starting day trading strategy")
    
    # Initialize components
    try:
        # Market data provider
        market_data = MarketDataProvider(api_key=config['market_data'].get('api_key'))
        
        # News data provider
        news_data = NewsDataProvider(api_key=config['market_data'].get('api_key'))
        
        # Stock scanner
        scanner = StockScanner(market_data, news_data)
        
        # Condition tracker
        condition_tracker = ConditionTracker(market_data, news_data)
        
        # Risk manager
        initial_capital = args.capital if args.capital else config['trading']['initial_capital']
        max_risk_per_trade_pct = args.max_risk if args.max_risk else config['trading']['max_risk_per_trade_pct']
        
        risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            daily_max_loss_pct=config['trading']['daily_max_loss_pct'],
            profit_loss_ratio=config['trading']['profit_loss_ratio'],
            max_open_positions=config['trading']['max_open_positions']
        )
        
        # Determine if running in simulation mode
        simulate = args.simulate or config['simulation'].get('enabled', True)
        
        # Trade manager
        # In a real implementation, we would initialize a broker API client here
        broker_api = None  # Placeholder for broker API client
        
        trade_manager = TradeManager(
            market_data_provider=market_data,
            scanner=scanner,
            risk_manager=risk_manager,
            condition_tracker=condition_tracker,
            broker_api=broker_api if not simulate else None
        )
        
        # Set trading parameters
        trade_manager.min_price = config['trading']['min_price']
        trade_manager.max_price = config['trading']['max_price']
        trade_manager.min_gap_pct = config['trading']['min_gap_pct']
        trade_manager.min_rel_volume = config['trading']['min_rel_volume']
        trade_manager.max_float = config['trading']['max_float']
        
        # Log initialization details
        logger.info(f"Strategy initialized with {initial_capital:.2f} capital")
        logger.info(f"Maximum risk per trade: {max_risk_per_trade_pct:.2f}%")
        logger.info(f"Running in {'simulation' if simulate else 'live trading'} mode")
        
        # Start trading session
        if not args.scan_only:
            trade_manager.start_trading_session()
        
        # Main trading loop
        run_trading_loop(trade_manager, scanner, condition_tracker, 
                       trade_logger, performance_logger, args.scan_only)
        
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Error running strategy: {e}", exc_info=True)
    finally:
        # Clean up
        if not args.scan_only and 'trade_manager' in locals():
            trade_manager.stop_trading_session()
        
        logger.info("Strategy shutdown complete")


def run_trading_loop(trade_manager, scanner, condition_tracker, 
                   trade_logger, performance_logger, scan_only=False):
    """
    Run the main trading loop.
    
    Parameters:
    -----------
    trade_manager: TradeManager
        Trading manager
    scanner: StockScanner
        Stock scanner
    condition_tracker: ConditionTracker
        Market condition tracker
    trade_logger: TradeLogger
        Trade logger
    performance_logger: PerformanceLogger
        Performance logger
    scan_only: bool
        Only scan for opportunities without trading
    """
    logger = logging.getLogger('trading_strategy')
    
    # Update market session
    condition_tracker.update_market_session()
    
    # Check if market is open
    if not (condition_tracker.market_open or condition_tracker.pre_market):
        logger.warning("Market is not open, running in simulation mode")
    
    # Main loop
    try:
        while True:
            # Update market conditions
            condition_tracker.update_market_conditions()
            
            # Scan for trading opportunities
            opportunities = trade_manager.scan_for_opportunities()
            
            if opportunities:
                logger.info(f"Found {len(opportunities)} trading opportunities")
                
                # Process each opportunity
                for stock in opportunities:
                    logger.info(f"Evaluating {stock.symbol} ({stock.current_price:.2f})")
                    
                    # Log opportunity details
                    opportunity_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': stock.symbol,
                        'price': stock.current_price,
                        'gap_percent': stock.gap_percent,
                        'relative_volume': stock.relative_volume,
                        'float': stock.shares_float,
                        'has_bull_flag': stock.has_bull_flag,
                        'has_micro_pullback': stock.has_micro_pullback,
                        'has_new_high_breakout': stock.has_new_high_breakout,
                        'news_headline': getattr(stock, 'news_headline', ''),
                        'scan_only': scan_only
                    }
                    
                    trade_logger.log_trade(opportunity_data)
                    
                    # If scan-only mode, continue to next opportunity
                    if scan_only:
                        continue
                    
                    # Evaluate the opportunity
                    trade_params = trade_manager.evaluate_opportunity(stock)
                    
                    if trade_params:
                        # Execute trade
                        executed_trade = trade_manager.execute_trade(trade_params)
                        
                        if executed_trade:
                            logger.info(f"Trade executed: {stock.symbol} - {executed_trade['executed_shares']} shares at ${executed_trade['executed_price']:.2f}")
                            
                            # Log trade details
                            trade_data = {
                                'timestamp': datetime.now().isoformat(),
                                'symbol': stock.symbol,
                                'action': 'buy',
                                'shares': executed_trade['executed_shares'],
                                'price': executed_trade['executed_price'],
                                'stop_price': executed_trade['stop_price'],
                                'target_price': executed_trade['target_price'],
                                'pattern': executed_trade['pattern']
                            }
                            
                            trade_logger.log_trade(trade_data)
            
            # Manage active trades
            if not scan_only:
                actions = trade_manager.manage_active_trades()
                
                for action in actions:
                    # Log trade management actions
                    action_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': action['symbol'],
                        'action': action['action'],
                        'reason': action['reason']
                    }
                    
                    if 'price' in action:
                        action_data['price'] = action['price']
                    
                    if 'pnl' in action:
                        action_data['pnl'] = action['pnl']
                    
                    trade_logger.log_trade(action_data)
            
            # Check if we should continue trading
            if not scan_only:
                should_continue, reason = trade_manager.risk_manager.should_continue_trading()
                
                if not should_continue:
                    logger.info(f"Stopping trading: {reason}")
                    break
            
            # Get trading status
            status = trade_manager.get_trading_status()
            
            # Log performance
            if not scan_only:
                performance_data = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'trades_taken': trade_manager.daily_stats['trades_taken'],
                    'winning_trades': trade_manager.daily_stats['winning_trades'],
                    'losing_trades': trade_manager.daily_stats['losing_trades'],
                    'win_rate': (trade_manager.daily_stats['winning_trades'] / 
                               trade_manager.daily_stats['trades_taken'] 
                               if trade_manager.daily_stats['trades_taken'] > 0 else 0),
                    'profit': trade_manager.daily_stats['total_profit'],
                    'loss': abs(trade_manager.daily_stats['total_loss']),
                    'net_pnl': trade_manager.daily_stats['gross_pnl'],
                    'cushion_achieved': status['cushion_achieved']
                }
                
                performance_logger.log_daily_performance(performance_data)
            
            # Wait for the next update
            logger.debug("Waiting for next update...")
            time.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        logger.info("Trading loop interrupted by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run strategy
    run_strategy(config, args)