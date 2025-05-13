"""
Triple Candlestick Patterns

These patterns are based on three consecutive candlesticks and their relationships.
"""

from .morning_star import MorningStarPattern
from .evening_star import EveningStarPattern
from .three_white_soldiers import ThreeWhiteSoldiersPattern
from .three_black_crows import ThreeBlackCrowsPattern

__all__ = [
    'MorningStarPattern',
    'EveningStarPattern',
    'ThreeWhiteSoldiersPattern',
    'ThreeBlackCrowsPattern'
]