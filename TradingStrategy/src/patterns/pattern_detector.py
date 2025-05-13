"""
Pattern Detector

Main pattern detection engine that orchestrates all candlestick patterns
and provides a unified interface for pattern detection.
"""

import logging
import importlib
import inspect
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from .base_pattern import BasePattern

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Main pattern detection engine that manages and executes all pattern detections.
    """
    
    def __init__(self, patterns_to_load: Optional[List[str]] = None):
        """
        Initialize the pattern detector.
        
        Parameters:
        -----------
        patterns_to_load: List[str], optional
            Specific patterns to load. If None, loads all available patterns.
        """
        self.patterns = {}
        self.pattern_types = {
            'single': {},
            'double': {},
            'triple': {},
            'complex': {}
        }
        
        # Load patterns
        if patterns_to_load:
            self._load_specific_patterns(patterns_to_load)
        else:
            self._load_all_patterns()
            
        logger.info(f"Pattern Detector initialized with {len(self.patterns)} patterns")
    
    def _load_all_patterns(self):
        """Load all available patterns from the patterns directory."""
        patterns_dir = Path(__file__).parent
        
        # Define subdirectories to scan
        subdirs = {
            'single_patterns': 'single',
            'double_patterns': 'double',
            'triple_patterns': 'triple',
            'complex_patterns': 'complex'
        }
        
        for subdir, pattern_type in subdirs.items():
            dir_path = patterns_dir / subdir
            if not dir_path.exists():
                logger.warning(f"Pattern directory not found: {dir_path}")
                continue
                
            # Find all .py files in the directory (except __init__.py)
            pattern_files = [f for f in dir_path.glob('*.py') if f.name != '__init__.py']
            
            for file_path in pattern_files:
                self._load_pattern_from_file(file_path, pattern_type)
    
    def _load_pattern_from_file(self, file_path: Path, pattern_type: str):
        """
        Load a pattern from a specific file.
        
        Parameters:
        -----------
        file_path: Path
            Path to the pattern file
        pattern_type: str
            Type of pattern (single, double, triple, complex)
        """
        try:
            # Convert file path to module path
            module_path = str(file_path.relative_to(Path(__file__).parent.parent))
            module_path = module_path.replace('/', '.').replace('\\', '.')[:-3]  # Remove .py
            module_path = 'src.patterns.' + module_path
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find classes that inherit from BasePattern
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BasePattern) and obj != BasePattern:
                    pattern_instance = obj()
                    pattern_name = pattern_instance.name.lower().replace(' ', '_')
                    
                    # Store in both the main dict and type-specific dict
                    self.patterns[pattern_name] = pattern_instance
                    self.pattern_types[pattern_type][pattern_name] = pattern_instance
                    
                    logger.info(f"Loaded pattern: {pattern_instance.name} ({pattern_type})")
                    
        except Exception as e:
            logger.error(f"Error loading pattern from {file_path}: {e}")
    
    def _load_specific_patterns(self, pattern_names: List[str]):
        """
        Load only specific patterns by name.
        
        Parameters:
        -----------
        pattern_names: List[str]
            List of pattern names to load
        """
        # First load all patterns
        self._load_all_patterns()
        
        # Filter to keep only requested patterns
        requested_patterns = {}
        requested_pattern_types = {
            'single': {},
            'double': {},
            'triple': {},
            'complex': {}
        }
        
        for pattern_name in pattern_names:
            normalized_name = pattern_name.lower().replace(' ', '_')
            if normalized_name in self.patterns:
                pattern = self.patterns[normalized_name]
                requested_patterns[normalized_name] = pattern
                
                # Add to type-specific dict
                pattern_type = pattern.pattern_type
                requested_pattern_types[pattern_type][normalized_name] = pattern
            else:
                logger.warning(f"Requested pattern not found: {pattern_name}")
        
        # Update with filtered patterns
        self.patterns = requested_patterns
        self.pattern_types = requested_pattern_types
    
    def detect_all_patterns(self, candles: pd.DataFrame, pattern_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect all patterns in the candlestick data.
        
        Parameters:
        -----------
        candles: pd.DataFrame
            OHLCV candlestick data
        pattern_types: List[str], optional
            Types of patterns to detect. If None, detects all types.
            Options: ['single', 'double', 'triple', 'complex']
            
        Returns:
        --------
        List[Dict]: List of detected patterns with their details
        """
        detected_patterns = []
        
        # Determine which pattern types to check
        if pattern_types is None:
            types_to_check = ['single', 'double', 'triple', 'complex']
        else:
            types_to_check = pattern_types
        
        # Check each pattern type
        for pattern_type in types_to_check:
            if pattern_type in self.pattern_types:
                patterns_of_type = self.pattern_types[pattern_type]
                
                for pattern_name, pattern in patterns_of_type.items():
                    try:
                        result = pattern.detect(candles)
                        if result:
                            result['pattern_type'] = pattern_type
                            detected_patterns.append(result)
                    except Exception as e:
                        logger.error(f"Error detecting {pattern_name}: {e}")
        
        # Sort by confidence (highest first)
        detected_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return detected_patterns
    
    def detect_specific_pattern(self, pattern_name: str, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Detect a specific pattern by name.
        
        Parameters:
        -----------
        pattern_name: str
            Name of the pattern to detect
        candles: pd.DataFrame
            OHLCV candlestick data
            
        Returns:
        --------
        Dict or None: Pattern detection result if found
        """
        normalized_name = pattern_name.lower().replace(' ', '_')
        
        if normalized_name not in self.patterns:
            logger.error(f"Pattern not found: {pattern_name}")
            return None
        
        pattern = self.patterns[normalized_name]
        
        try:
            result = pattern.detect(candles)
            if result:
                result['pattern_type'] = pattern.pattern_type
            return result
        except Exception as e:
            logger.error(f"Error detecting {pattern_name}: {e}")
            return None
    
    def get_bullish_patterns(self, candles: pd.DataFrame) -> List[Dict]:
        """
        Detect only bullish patterns.
        
        Parameters:
        -----------
        candles: pd.DataFrame
            OHLCV candlestick data
            
        Returns:
        --------
        List[Dict]: List of detected bullish patterns
        """
        all_patterns = self.detect_all_patterns(candles)
        return [p for p in all_patterns if p.get('direction') == 'bullish']
    
    def get_bearish_patterns(self, candles: pd.DataFrame) -> List[Dict]:
        """
        Detect only bearish patterns.
        
        Parameters:
        -----------
        candles: pd.DataFrame
            OHLCV candlestick data
            
        Returns:
        --------
        List[Dict]: List of detected bearish patterns
        """
        all_patterns = self.detect_all_patterns(candles)
        return [p for p in all_patterns if p.get('direction') == 'bearish']
    
    def get_high_confidence_patterns(self, candles: pd.DataFrame, min_confidence: float = 70.0) -> List[Dict]:
        """
        Detect only high confidence patterns.
        
        Parameters:
        -----------
        candles: pd.DataFrame
            OHLCV candlestick data
        min_confidence: float
            Minimum confidence threshold (0-100)
            
        Returns:
        --------
        List[Dict]: List of high confidence patterns
        """
        all_patterns = self.detect_all_patterns(candles)
        return [p for p in all_patterns if p.get('confidence', 0) >= min_confidence]
    
    def get_patterns_for_trend(self, candles: pd.DataFrame, trend: str) -> List[Dict]:
        """
        Get patterns appropriate for a specific trend.
        
        Parameters:
        -----------
        candles: pd.DataFrame
            OHLCV candlestick data
        trend: str
            Trend type ('uptrend', 'downtrend', 'sideways')
            
        Returns:
        --------
        List[Dict]: List of patterns suitable for the trend
        """
        if trend == 'uptrend':
            # In uptrend, look for bullish continuation and bearish reversal patterns
            patterns = self.detect_all_patterns(candles)
            return [p for p in patterns if 
                    (p.get('direction') == 'bullish' and p.get('pattern_type') in ['complex']) or
                    (p.get('direction') == 'bearish' and p.get('pattern_type') in ['single', 'double', 'triple'])]
        
        elif trend == 'downtrend':
            # In downtrend, look for bearish continuation and bullish reversal patterns
            patterns = self.detect_all_patterns(candles)
            return [p for p in patterns if 
                    (p.get('direction') == 'bearish' and p.get('pattern_type') in ['complex']) or
                    (p.get('direction') == 'bullish' and p.get('pattern_type') in ['single', 'double', 'triple'])]
        
        else:  # sideways
            # In sideways market, all patterns are potentially valid
            return self.detect_all_patterns(candles)
    
    def get_available_patterns(self) -> Dict[str, List[str]]:
        """
        Get a list of all available patterns organized by type.
        
        Returns:
        --------
        Dict[str, List[str]]: Dictionary with pattern types as keys and pattern names as values
        """
        available = {}
        for pattern_type, patterns in self.pattern_types.items():
            available[pattern_type] = [p.name for p in patterns.values()]
        return available
    
    def get_pattern_info(self, pattern_name: str) -> Optional[Dict]:
        """
        Get information about a specific pattern.
        
        Parameters:
        -----------
        pattern_name: str
            Name of the pattern
            
        Returns:
        --------
        Dict or None: Pattern information if found
        """
        normalized_name = pattern_name.lower().replace(' ', '_')
        
        if normalized_name not in self.patterns:
            return None
        
        pattern = self.patterns[normalized_name]
        
        return {
            'name': pattern.name,
            'type': pattern.pattern_type,
            'min_candles_required': pattern.min_candles_required,
            'description': pattern.__doc__,
            'module': pattern.__class__.__module__
        }