"""
Complex Trading Patterns

These are Ross Cameron's main momentum trading patterns.
"""

from .bull_flag import BullFlagPattern
from .bull_pennant import BullPennantPattern
from .first_pullback import FirstPullbackPattern
from .micro_pullback import MicroPullbackPattern
from .flat_top_breakout import FlatTopBreakoutPattern
from .candle_over_candle import CandleOverCandlePattern
from .new_high_breakout import NewHighBreakoutPattern

__all__ = [
    'BullFlagPattern',
    'BullPennantPattern',
    'FirstPullbackPattern',
    'MicroPullbackPattern',
    'FlatTopBreakoutPattern',
    'CandleOverCandlePattern',
    'NewHighBreakoutPattern'
]