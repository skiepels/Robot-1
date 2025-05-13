# New file: src/conditions/condition_checker.py
"""
Central Condition Checker

Monitors all 5 conditions for trading strategy
"""

from .condition1_price import PriceCondition
from .condition2_percent_up import PercentUpCondition
from .condition3_volume import VolumeCondition
from .condition4_news import NewsCondition
from .condition5_float import FloatCondition

class ConditionChecker:
    def __init__(self):
        # Initialize all conditions parameters
        self.price_condition = PriceCondition(min_price=2.0, max_price=20.0)
        self.percent_up_condition = PercentUpCondition(min_percent=10.0)
        self.volume_condition = VolumeCondition(min_rel_volume=5.0)
        self.news_condition = NewsCondition(max_hours_old=24)
        self.float_condition = FloatCondition(max_float=20_000_000)
    
    def check_all_conditions(self, stock_data):
        """
        Check if a stock meets all 5 conditions.
        
        Parameters:
        -----------
        stock_data: dict
            Contains all necessary stock information
            
        Returns:
        --------
        tuple: (bool, dict) - Whether all conditions are met and details
        """
        results = {
            'price': self.price_condition.check(stock_data.get('current_price')),
            'percent_up': self.percent_up_condition.check(stock_data.get('day_change_percent')),
            'volume': self.volume_condition.check(stock_data.get('relative_volume')),
            'news': self.news_condition.check(stock_data.get('has_news')),
            'float': self.float_condition.check(stock_data.get('shares_float'))
        }
        
        all_conditions_met = all(results.values())
        
        return all_conditions_met, results
    
    def get_condition_status(self, stock_data):
        """Get detailed status of each condition."""
        return {
            'price': {
                'met': self.price_condition.check(stock_data.get('current_price')),
                'value': stock_data.get('current_price'),
                'requirement': self.price_condition.get_description()
            },
            'percent_up': {
                'met': self.percent_up_condition.check(stock_data.get('day_change_percent')),
                'value': stock_data.get('day_change_percent'),
                'requirement': self.percent_up_condition.get_description()
            },
            'volume': {
                'met': self.volume_condition.check(stock_data.get('relative_volume')),
                'value': stock_data.get('relative_volume'),
                'requirement': self.volume_condition.get_description()
            },
            'news': {
                'met': self.news_condition.check(stock_data.get('has_news')),
                'value': stock_data.get('has_news'),
                'requirement': self.news_condition.get_description()
            },
            'float': {
                'met': self.float_condition.check(stock_data.get('shares_float')),
                'value': stock_data.get('shares_float'),
                'requirement': self.float_condition.get_description()
            }
        }