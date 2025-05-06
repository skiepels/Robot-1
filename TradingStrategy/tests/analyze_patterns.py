"""
Pattern Detection Analysis

This script analyzes the accuracy of the candlestick pattern detection algorithms
by testing them against a set of known chart patterns and visualizing the results.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
from datetime import datetime, timedelta
import mplfinance as mpf

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.candlestick_patterns import CandlestickPatterns


class PatternAnalyzer:
    """Analyzer for testing and visualizing candlestick pattern detection."""
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.pattern_detector = CandlestickPatterns()
        
        # Create output directory
        os.makedirs('logs/pattern_analysis', exist_ok=True)
    
    def generate_sample_patterns(self):
        """
        Generate sample price data for various candlestick patterns.
        
        Returns:
        --------
        dict: Dictionary with pattern types as keys and corresponding DataFrames as values
        """
        patterns = {}
        
        # Generate dates for the patterns
        dates = pd.date_range(start='2025-05-01', periods=20, freq='1min')
        
        # 1. Bull Flag Pattern
        bull_flag_data = pd.DataFrame({
            'open': [10.0, 10.2, 10.5, 11.0, 11.5, 11.3, 11.2, 11.0, 10.9, 10.8,
                   10.9, 11.0, 11.2, 11.4, 11.7, 12.0, 12.2, 12.5, 12.8, 13.0],
            'high': [10.3, 10.4, 10.8, 11.3, 11.8, 11.5, 11.4, 11.2, 11.1, 11.0,
                    11.1, 11.2, 11.4, 11.6, 11.9, 12.2, 12.5, 12.7, 13.0, 13.2],
            'low': [9.9, 10.0, 10.4, 10.8, 11.3, 11.0, 10.9, 10.7, 10.6, 10.5,
                  10.7, 10.8, 11.0, 11.2, 11.5, 11.8, 12.0, 12.3, 12.6, 12.8],
            'close': [10.2, 10.5, 10.7, 11.2, 11.7, 11.2, 11.0, 10.9, 10.8, 10.7,
                    11.0, 11.2, 11.4, 11.6, 11.8, 12.1, 12.4, 12.6, 12.9, 13.1],
            'volume': [5000, 8000, 10000, 15000, 20000, 15000, 12000, 10000, 8000, 7000,
                     9000, 11000, 13000, 15000, 18000, 22000, 25000, 20000, 18000, 15000]
        }, index=dates)
        
        # Add VWAP
        bull_flag_data['vwap'] = self._calculate_vwap(bull_flag_data)
        
        patterns['bull_flag'] = bull_flag_data
        
        # 2. Micro Pullback Pattern
        micro_pullback_data = pd.DataFrame({
            'open': [10.0, 10.2, 10.5, 10.8, 11.0, 11.3, 11.5, 11.8, 11.7, 12.0,
                   12.2, 12.5, 12.8, 13.0, 13.2, 13.5, 13.8, 14.0, 14.2, 14.5],
            'high': [10.2, 10.4, 10.7, 11.0, 11.2, 11.5, 11.8, 12.0, 11.9, 12.2,
                    12.4, 12.7, 13.0, 13.3, 13.5, 13.8, 14.0, 14.3, 14.5, 14.8],
            'low': [9.8, 10.0, 10.3, 10.6, 10.8, 11.1, 11.3, 11.6, 11.5, 11.7,
                  12.0, 12.3, 12.5, 12.8, 13.0, 13.3, 13.6, 13.8, 14.0, 14.3],
            'close': [10.1, 10.3, 10.6, 10.9, 11.1, 11.4, 11.7, 11.9, 11.6, 12.1,
                    12.3, 12.6, 12.9, 13.1, 13.4, 13.7, 13.9, 14.2, 14.4, 14.7],
            'volume': [5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 9000, 14000,
                     16000, 18000, 20000, 22000, 24000, 20000, 18000, 16000, 14000, 12000]
        }, index=dates)
        
        # Add VWAP
        micro_pullback_data['vwap'] = self._calculate_vwap(micro_pullback_data)
        
        patterns['micro_pullback'] = micro_pullback_data
        
        # 3. New High Breakout Pattern
        new_high_data = pd.DataFrame({
            'open': [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.0, 10.8, 10.6,
                   10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.3, 12.5, 12.8],
            'high': [10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.2, 11.0, 10.8,
                    11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.5, 12.7, 13.0],
            'low': [9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 10.8, 10.6, 10.4,
                  10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.1, 12.3, 12.6],
            'close': [10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 10.9, 10.7, 10.5,
                    10.9, 11.1, 11.3, 11.5, 11.7, 11.9, 12.1, 12.4, 12.6, 12.9],
            'volume': [5000, 5500, 6000, 6500, 7000, 7500, 8000, 6000, 5500, 5000,
                     6000, 7000, 8000, 9000, 10000, 11000, 12000, 15000, 16000, 18000]
        }, index=dates)
        
        # Add VWAP
        new_high_data['vwap'] = self._calculate_vwap(new_high_data)
        
        patterns['new_high_breakout'] = new_high_data
        
        # 4. Doji Pattern
        doji_data = pd.DataFrame({
            'open': [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.5, 11.5,
                   11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4],
            'high': [10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 11.8,
                    11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6],
            'low': [9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.2, 11.2,
                  11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2],
            'close': [10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.5, 11.5,
                    11.7, 11.9, 12.1, 12.3, 12.5, 12.7, 12.9, 13.1, 13.3, 13.5],
            'volume': [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,
                     10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500]
        }, index=dates)
        
        # Add VWAP
        doji_data['vwap'] = self._calculate_vwap(doji_data)
        
        # Make the 9th candle a doji
        doji_data.loc[doji_data.index[8], 'open'] = 11.50
        doji_data.loc[doji_data.index[8], 'close'] = 11.51
        doji_data.loc[doji_data.index[8], 'high'] = 11.80
        doji_data.loc[doji_data.index[8], 'low'] = 11.20
        
        patterns['doji'] = doji_data
        
        # 5. Hammer Pattern
        hammer_data = pd.DataFrame({
            'open': [10.0, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6, 8.8, 9.0,
                   9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0],
            'high': [10.2, 10.0, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 9.0, 9.2,
                    9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
            'low': [9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6, 8.0, 8.6, 8.8,
                  9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8],
            'close': [9.9, 9.7, 9.5, 9.3, 9.1, 8.9, 8.7, 8.8, 8.9, 9.1,
                    9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1],
            'volume': [5000, 4800, 4600, 4400, 4200, 4000, 3800, 3600, 5000, 5200,
                     5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200]
        }, index=dates)
        
        # Add VWAP
        hammer_data['vwap'] = self._calculate_vwap(hammer_data)
        
        # Make the 7th candle a hammer
        hammer_data.loc[hammer_data.index[7], 'open'] = 8.65
        hammer_data.loc[hammer_data.index[7], 'close'] = 8.75
        hammer_data.loc[hammer_data.index[7], 'high'] = 8.78
        hammer_data.loc[hammer_data.index[7], 'low'] = 8.0
        
        patterns['hammer'] = hammer_data
        
        return patterns
    
    def _calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def detect_patterns(self, patterns_dict):
        """
        Detect patterns in the sample data.
        
        Parameters:
        -----------
        patterns_dict: dict
            Dictionary with pattern types as keys and DataFrames as values
            
        Returns:
        --------
        dict: Dictionary with pattern types and detection results
        """
        results = {}
        
        for pattern_type, df in patterns_dict.items():
            if pattern_type == 'bull_flag':
                detected = self.pattern_detector.detect_bull_flag(df)
                results[pattern_type] = detected
                
            elif pattern_type == 'micro_pullback':
                detected = self.pattern_detector.detect_micro_pullback(df)
                results[pattern_type] = detected
                
            elif pattern_type == 'new_high_breakout':
                detected = self.pattern_detector.detect_first_candle_to_make_new_high(df)
                results[pattern_type] = detected
                
            elif pattern_type == 'doji':
                detected = [i for i in range(len(df)) 
                           if self.pattern_detector.detect_doji(df, i)]
                results[pattern_type] = detected
                
            elif pattern_type == 'hammer':
                detected = [i for i in range(len(df)) 
                           if self.pattern_detector.detect_hammer(df, i)]
                results[pattern_type] = detected
        
        return results
    
    def visualize_patterns(self, patterns_dict, detection_results):
        """
        Visualize the patterns and detection results.
        
        Parameters:
        -----------
        patterns_dict: dict
            Dictionary with pattern types as keys and DataFrames as values
        detection_results: dict
            Dictionary with pattern types and detection results
        """
        for pattern_type, df in patterns_dict.items():
            # Convert DataFrame to mplfinance format
            df_mpf = df.copy()
            df_mpf.columns = [col.capitalize() for col in df_mpf.columns]
            
            # Prepare figure and adding detection markers
            detected_indices = detection_results[pattern_type]
            
            # Create the candlestick chart
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Format the dates for the x-axis
            date_format = DateFormatter('%H:%M')
            axes[0].xaxis.set_major_formatter(date_format)
            
            # Plot candlesticks
            mpf.plot(df_mpf, type='candle', ax=axes[0], volume=axes[1], show_nontrading=False)
            
            # Add VWAP line
            axes[0].plot(df.index, df['vwap'], label='VWAP', color='purple', linewidth=1)
            
            # Mark detected patterns
            if detected_indices:
                for idx in detected_indices:
                    x_date = df.index[idx]
                    y_price = df['low'].iloc[idx] * 0.98  # Place below the candle
                    
                    # Add marker and annotation
                    axes[0].scatter(x_date, y_price, color='red', marker='^', s=100)
                    axes[0].annotate('Detected', (x_date, y_price), 
                                  xytext=(0, -20), textcoords='offset points',
                                  ha='center', color='red', fontweight='bold')
            
            # Add title and legend
            pattern_name = pattern_type.replace('_', ' ').title()
            axes[0].set_title(f'{pattern_name} Pattern', fontsize=14)
            axes[0].legend()
            
            # Save figure
            plt.tight_layout()
            fig.savefig(f'logs/pattern_analysis/{pattern_type}.png')
            plt.close(fig)
            
            print(f"Visualization saved for {pattern_type} pattern")
    
    def test_pattern_robustness(self, pattern_type, variations=5):
        """
        Test the robustness of pattern detection by adding noise to the data.
        
        Parameters:
        -----------
        pattern_type: str
            The type of pattern to test
        variations: int
            Number of variations to test
            
        Returns:
        --------
        dict: Dictionary with noise levels and detection success
        """
        # Get the base pattern
        patterns = self.generate_sample_patterns()
        base_df = patterns[pattern_type]
        
        # Noise levels to test
        noise_levels = np.linspace(0.01, 0.10, variations)  # 1% to 10% noise
        
        results = {}
        
        for noise_level in noise_levels:
            successes = 0
            
            # Run multiple trials for each noise level
            for trial in range(10):
                # Add random noise to the data
                noisy_df = base_df.copy()
                
                # Add noise to OHLC
                for col in ['open', 'high', 'low', 'close']:
                    noise = np.random.normal(0, noise_level, len(noisy_df))
                    noisy_df[col] = noisy_df[col] * (1 + noise)
                
                # Ensure high >= open, close and low <= open, close
                for idx in range(len(noisy_df)):
                    noisy_df.loc[noisy_df.index[idx], 'high'] = max(
                        noisy_df.loc[noisy_df.index[idx], 'high'],
                        noisy_df.loc[noisy_df.index[idx], 'open'],
                        noisy_df.loc[noisy_df.index[idx], 'close']
                    )
                    
                    noisy_df.loc[noisy_df.index[idx], 'low'] = min(
                        noisy_df.loc[noisy_df.index[idx], 'low'],
                        noisy_df.loc[noisy_df.index[idx], 'open'],
                        noisy_df.loc[noisy_df.index[idx], 'close']
                    )
                
                # Recalculate VWAP
                noisy_df['vwap'] = self._calculate_vwap(noisy_df)
                
                # Detect pattern
                if pattern_type == 'bull_flag':
                    detected = self.pattern_detector.detect_bull_flag(noisy_df)
                elif pattern_type == 'micro_pullback':
                    detected = self.pattern_detector.detect_micro_pullback(noisy_df)
                elif pattern_type == 'new_high_breakout':
                    detected = self.pattern_detector.detect_first_candle_to_make_new_high(noisy_df)
                elif pattern_type == 'doji':
                    detected = [i for i in range(len(noisy_df)) 
                              if self.pattern_detector.detect_doji(noisy_df, i)]
                elif pattern_type == 'hammer':
                    detected = [i for i in range(len(noisy_df)) 
                              if self.pattern_detector.detect_hammer(noisy_df, i)]
                
                # Check if any pattern was detected
                if detected:
                    successes += 1
            
            # Calculate success rate
            success_rate = successes / 10.0
            results[noise_level] = success_rate
        
        return results
    
    def visualize_robustness_results(self, robustness_results):
        """
        Visualize the pattern detection robustness results.
        
        Parameters:
        -----------
        robustness_results: dict
            Dictionary with pattern types and their robustness results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for pattern_type, results in robustness_results.items():
            # Extract noise levels and success rates
            noise_levels = list(results.keys())
            success_rates = list(results.values())
            
            # Plot the results
            pattern_name = pattern_type.replace('_', ' ').title()
            ax.plot(noise_levels, success_rates, marker='o', label=pattern_name)
        
        # Add labels and title
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Detection Success Rate')
        ax.set_title('Pattern Detection Robustness')
        ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        fig.savefig('logs/pattern_analysis/robustness_results.png')
        plt.close(fig)
        
        print("Robustness visualization saved")
    
    def analyze_all_patterns(self):
        """Analyze all patterns and generate reports."""
        print("Generating sample patterns...")
        patterns = self.generate_sample_patterns()
        
        print("Detecting patterns...")
        detection_results = self.detect_patterns(patterns)
        
        print("Visualizing patterns...")
        self.visualize_patterns(patterns, detection_results)
        
        print("Testing pattern robustness...")
        robustness_results = {}
        for pattern_type in patterns.keys():
            print(f"Testing robustness of {pattern_type} pattern...")
            robustness_results[pattern_type] = self.test_pattern_robustness(pattern_type)
        
        print("Visualizing robustness results...")
        self.visualize_robustness_results(robustness_results)
        
        # Generate summary report
        self._generate_summary_report(patterns, detection_results, robustness_results)
        
        print("Analysis complete. Results saved to logs/pattern_analysis/")
    
    def _generate_summary_report(self, patterns, detection_results, robustness_results):
        """
        Generate a summary report of the pattern analysis.
        
        Parameters:
        -----------
        patterns: dict
            Dictionary with pattern types as keys and DataFrames as values
        detection_results: dict
            Dictionary with pattern types and detection results
        robustness_results: dict
            Dictionary with pattern types and their robustness results
        """
        with open('logs/pattern_analysis/summary_report.txt', 'w') as f:
            f.write("=== Pattern Analysis Summary Report ===\n\n")
            
            # Overall detection results
            f.write("Pattern Detection Results:\n")
            f.write("--------------------------\n")
            for pattern_type, detected in detection_results.items():
                pattern_name = pattern_type.replace('_', ' ').title()
                if detected:
                    f.write(f"{pattern_name}: Detected at indices {detected}\n")
                else:
                    f.write(f"{pattern_name}: Not detected\n")
            
            f.write("\n")
            
            # Robustness results
            f.write("Pattern Robustness Results:\n")
            f.write("--------------------------\n")
            for pattern_type, results in robustness_results.items():
                pattern_name = pattern_type.replace('_', ' ').title()
                f.write(f"{pattern_name} Pattern:\n")
                
                for noise_level, success_rate in results.items():
                    f.write(f"  Noise Level {noise_level:.2f}: {success_rate:.2f} success rate\n")
                
                # Calculate average success rate
                avg_success = sum(results.values()) / len(results)
                f.write(f"  Average Success Rate: {avg_success:.2f}\n\n")
            
            # Overall assessment
            f.write("Overall Assessment:\n")
            f.write("------------------\n")
            
            # Calculate average robustness for each pattern
            avg_robustness = {}
            for pattern_type, results in robustness_results.items():
                avg_robustness[pattern_type] = sum(results.values()) / len(results)
            
            # Sort patterns by robustness
            sorted_patterns = sorted(avg_robustness.items(), key=lambda x: x[1], reverse=True)
            
            for pattern_type, avg_score in sorted_patterns:
                pattern_name = pattern_type.replace('_', ' ').title()
                
                if avg_score > 0.8:
                    assessment = "Excellent"
                elif avg_score > 0.6:
                    assessment = "Good"
                elif avg_score > 0.4:
                    assessment = "Fair"
                else:
                    assessment = "Poor"
                
                f.write(f"{pattern_name} Pattern: {assessment} robustness (Score: {avg_score:.2f})\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("---------------\n")
            
            for pattern_type, avg_score in sorted_patterns:
                pattern_name = pattern_type.replace('_', ' ').title()
                
                if avg_score < 0.6:
                    f.write(f"- Improve {pattern_name} pattern detection robustness\n")
            
            f.write("\n")
            
            # Ross Cameron's approach
            f.write("Ross Cameron's Approach:\n")
            f.write("----------------------\n")
            f.write("Ross Cameron's trading strategy relies heavily on accurate pattern detection.\n")
            f.write("Key patterns in his approach:\n")
            f.write("1. Bull Flag - Used for momentum continuation trades\n")
            f.write("2. Micro Pullback - Used for quick entries on small retracements\n")
            f.write("3. First Candle to Make a New High - Primary breakout entry pattern\n")
            f.write("\n")
            f.write("The pattern detection algorithms should prioritize these patterns\n")
            f.write("and ensure they are robust against market noise and variations.\n")


if __name__ == '__main__':
    analyzer = PatternAnalyzer()
    analyzer.analyze_all_patterns()