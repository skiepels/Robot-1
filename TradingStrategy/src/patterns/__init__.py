"""
Pattern Detection Module

This module provides all the pattern detection functionality
for the trading strategy.
"""

from .pattern_detector import PatternDetector
from .base_pattern import BasePattern

# Import all pattern types
from .single_patterns import *
from .double_patterns import *
from .triple_patterns import *
from .complex_patterns import *

__all__ = [
    'PatternDetector',
    'BasePattern'
]