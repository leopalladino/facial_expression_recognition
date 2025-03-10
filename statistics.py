"""
Statistics module for the facial expression recognition system.
Generates charts and visualizations of expression data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import Counter
import logging
import config
from datetime import datetime

# Set up logging
logger = logging.getLogger("facial_expression")

class ExpressionStatistics:
    """Class for analyzing and visualizing expression data."""
    
    def __init__(self, output_dir: Optional[str] = None, detector_backend: str = "default"):
        """
        Initialize the expression statistics analyzer.

        Args:
            output_dir: Directory to save output charts (default: current directory)
            detector_backend: Name of the detector backend used
        """
        # Create a directory structure based on the detector backend
        if output_dir:
            self.base_output_dir = output_dir
        else:
            self.base_output_dir = os.path.join(config.BASE_DIR, "statistics")

        # Create detector-specific directory
        self.output_dir = os.path.join(self.base_output_dir, detector_backend)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize data structures
        self.expressions_data = []  # List of (frame_number, expression, confidence) tuples
        self.frame_count = 0
        self.detector_backend = detector_backend

        # Set color map for consistent colors across charts
        self.color_map = {
            "Angry": "#FF0000",      # Red
            "Disgust": "#FFA500",    # Orange
            "Fear": "#FFFF00",       # Yellow
            "Happy": "#00FF00",      # Green
            "Sad": "#0000FF",        # Blue
            "Surprise": "#FF00FF",   # Magenta
            "Neutral": "#FFFFFF",    # White
            "Unknown": "#808080"     # Gray
        }

        logger.info(f"Expression statistics initialized. Output directory: {self.output_dir}")

    def add_frame_data(self, expressions_with_confidence: List[Tuple[str, float]]):
        """
        Add expression data from a frame.

        Args:
            expressions_with_confidence: List of (expression, confidence) tuples
        """
        for expression, confidence in expressions_with_confidence:
            # Skip Neutral expressions as requested
            if expression != "Neutral":
                self.expressions_data.append((self.frame_count, expression, confidence))

        self.frame_count += 1

    def _prepare_data_for_analysis(self):
        """
        Prepare data for analysis by converting to DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with expression data
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.expressions_data, columns=['frame', 'expression', 'confidence'])

        # Handle empty data
        if df.empty:
            logger.warning("No expression data available for analysis")
            return pd.DataFrame()

        # Filter out Neutral expressions
        df = df[df['expression'] != "Neutral"]

        return df

    def generate_overall_expression_chart(self) -> str:
        """
        Generate a pie chart showing the overall distribution of expressions.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Count occurrences of each expression
        expression_counts = df['expression'].value_counts()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Create pie chart
        colors = [self.color_map.get(exp, "#808080") for exp in expression_counts.index]
        wedges, texts, autotexts = plt.pie(
            expression_counts,
            labels=expression_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )

        # Style the chart
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f'Overall Expression Distribution - {self.detector_backend}', fontsize=16)

        # Style the text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"expression_distribution_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Overall expression chart saved to {output_path}")
        return output_path

    def generate_expression_bar_chart(self) -> str:
        """
        Generate a bar chart showing the count of each expression.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Count occurrences of each expression
        expression_counts = df['expression'].value_counts()

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create bar chart
        colors = [self.color_map.get(exp, "#808080") for exp in expression_counts.index]
        bars = plt.bar(expression_counts.index, expression_counts.values, color=colors)

        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        # Style the chart
        plt.title(f'Expression Counts - {self.detector_backend}', fontsize=16)
        plt.xlabel('Expression', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"expression_counts_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Expression bar chart saved to {output_path}")
        return output_path

    def generate_confidence_flow_charts(self) -> List[str]:
        """
        Generate scatter plots showing the confidence flow for each expression over time.

        Returns:
            List[str]: Paths to the saved charts
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return []

        output_paths = []

        # Generate individual charts for each expression
        for expression in [e for e in config.EXPRESSIONS if e != "Neutral"] + ["Unknown"]:
            # Filter data for this expression
            expr_data = df[df['expression'] == expression]

            # Skip if no data for this expression
            if expr_data.empty:
                continue

            # Create figure
            plt.figure(figsize=(14, 8))

            # Create scatter plot instead of line chart
            plt.scatter(
                expr_data['frame'],
                expr_data['confidence'],
                color=self.color_map.get(expression, "#808080"),
                s=30,  # Marker size
                alpha=0.7,
                label=expression
            )

            # Style the chart
            plt.title(f'Confidence Flow for {expression} Expression - {self.detector_backend}', fontsize=16)
            plt.xlabel('Frame Number', fontsize=14)
            plt.ylabel('Confidence', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)  # Confidence is between 0 and 1

            # Add horizontal line at threshold
            plt.axhline(
                y=config.FACE_CONFIDENCE_THRESHOLD,
                color='r',
                linestyle='--',
                alpha=0.5,
                label=f'Threshold ({config.FACE_CONFIDENCE_THRESHOLD})'
            )

            plt.legend()

            # Adjust layout
            plt.tight_layout()

            # Save the chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"{expression}_confidence_flow_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Confidence flow chart for {expression} saved to {output_path}")
            output_paths.append(output_path)

        return output_paths

    def generate_combined_confidence_flow_chart(self) -> str:
        """
        Generate a scatter plot showing the confidence flow for all expressions over time.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Create figure
        plt.figure(figsize=(16, 10))

        # Get all expressions in the data
        expressions = df['expression'].unique()

        # Plot each expression
        for expression in expressions:
            # Filter data for this expression
            expr_data = df[df['expression'] == expression]

            # Skip if no data for this expression
            if expr_data.empty:
                continue

            # Create scatter plot instead of line chart
            plt.scatter(
                expr_data['frame'],
                expr_data['confidence'],
                color=self.color_map.get(expression, "#808080"),
                s=20,  # Marker size
                alpha=0.7,
                label=expression
            )

        # Style the chart
        plt.title(f'Combined Confidence Flow for All Expressions - {self.detector_backend}', fontsize=16)
        plt.xlabel('Frame Number', fontsize=14)
        plt.ylabel('Confidence', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)  # Confidence is between 0 and 1

        # Add horizontal line at threshold
        plt.axhline(
            y=config.FACE_CONFIDENCE_THRESHOLD,
            color='r',
            linestyle='--',
            alpha=0.5,
            label=f'Threshold ({config.FACE_CONFIDENCE_THRESHOLD})'
        )

        plt.legend(loc='upper right', fontsize=12)

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"combined_confidence_flow_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Combined confidence flow chart saved to {output_path}")
        return output_path

    def generate_expression_timeline_chart(self) -> str:
        """
        Generate a timeline chart showing which expression was dominant at each frame.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Group by frame and find the expression with highest confidence for each frame
        dominant_expressions = df.loc[df.groupby('frame')['confidence'].idxmax()]

        # Create figure
        plt.figure(figsize=(16, 8))

        # Create a colormap for the expressions
        unique_expressions = dominant_expressions['expression'].unique()
        colors = [self.color_map.get(exp, "#808080") for exp in unique_expressions]

        # Create a scatter plot with different colors for each expression
        for i, expression in enumerate(unique_expressions):
            mask = dominant_expressions['expression'] == expression
            plt.scatter(
                dominant_expressions[mask]['frame'],
                np.ones(mask.sum()),  # All points at y=1
                color=self.color_map.get(expression, "#808080"),
                label=expression,
                s=50,  # Marker size
                alpha=0.7
            )

        # Style the chart
        plt.title(f'Expression Timeline - {self.detector_backend}', fontsize=16)
        plt.xlabel('Frame Number', fontsize=14)
        plt.yticks([])  # Hide y-axis ticks
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(unique_expressions))

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"expression_timeline_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Expression timeline chart saved to {output_path}")
        return output_path

    def generate_confidence_heatmap(self) -> str:
        """
        Generate a heatmap showing the confidence distribution for each expression.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Create a pivot table with expressions as rows and confidence bins as columns
        # First, create confidence bins
        df['confidence_bin'] = pd.cut(df['confidence'], bins=10, labels=False)

        # Create pivot table
        pivot = pd.pivot_table(
            df,
            values='frame',
            index='expression',
            columns='confidence_bin',
            aggfunc='count',
            fill_value=0
        )

        # Create figure
        plt.figure(figsize=(14, 10))

        # Create heatmap
        im = plt.imshow(pivot, cmap='viridis')

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Count', rotation=270, labelpad=20)

        # Add labels
        plt.title(f'Expression Confidence Distribution - {self.detector_backend}', fontsize=16)
        plt.xlabel('Confidence Level (0-1 in bins)', fontsize=14)
        plt.ylabel('Expression', fontsize=14)

        # Set ticks
        plt.xticks(range(10), [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)])
        plt.yticks(range(len(pivot.index)), pivot.index)

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = plt.text(j, i, int(pivot.iloc[i, j]),
                               ha="center", va="center", color="w")

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"confidence_heatmap_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confidence heatmap saved to {output_path}")
        return output_path

    def generate_confidence_barplot_over_time(self) -> str:
        """
        Generate a bar plot showing confidence values over time with bars colored by emotion.

        Returns:
            str: Path to the saved chart
        """
        df = self._prepare_data_for_analysis()
        if df.empty:
            return ""

        # Create figure
        plt.figure(figsize=(16, 10))

        # Group by frame and get the highest confidence expression for each frame
        frame_data = df.loc[df.groupby('frame')['confidence'].idxmax()]

        # Sort by frame number to ensure chronological order
        frame_data = frame_data.sort_values('frame')

        # Create bar chart
        bars = plt.bar(
            frame_data['frame'],
            frame_data['confidence'],
            width=1.0,
            alpha=0.8
        )

        # Color each bar according to its expression
        for i, bar in enumerate(bars):
            expression = frame_data.iloc[i]['expression']
            bar.set_color(self.color_map.get(expression, "#808080"))

        # Add a legend
        unique_expressions = frame_data['expression'].unique()
        legend_handles = [plt.Rectangle((0,0),1,1, color=self.color_map.get(exp, "#808080")) for exp in unique_expressions]
        plt.legend(legend_handles, unique_expressions, loc='upper right')

        # Style the chart
        plt.title(f'Expression Confidence Over Time - {self.detector_backend}', fontsize=16)
        plt.xlabel('Frame Number', fontsize=14)
        plt.ylabel('Confidence', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line at threshold
        plt.axhline(
            y=config.FACE_CONFIDENCE_THRESHOLD,
            color='r',
            linestyle='--',
            alpha=0.5,
            label=f'Threshold ({config.FACE_CONFIDENCE_THRESHOLD})'
        )

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"confidence_barplot_over_time_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confidence barplot over time saved to {output_path}")
        return output_path

    def generate_all_charts(self) -> Dict[str, str]:
        """
        Generate all available charts.

        Returns:
            Dict[str, str]: Dictionary mapping chart types to file paths
        """
        logger.info(f"Generating all charts for {self.detector_backend}...")

        charts = {}

        # Generate expression bar chart
        charts['bar_chart'] = self.generate_expression_bar_chart()

        # Generate combined confidence flow chart
        charts['combined_flow_chart'] = self.generate_combined_confidence_flow_chart()

        # Generate confidence heatmap
        charts['heatmap'] = self.generate_confidence_heatmap()

        # Generate new confidence barplot over time
        charts['confidence_barplot_over_time'] = self.generate_confidence_barplot_over_time()

        logger.info(f"Generated {len(charts)} charts for {self.detector_backend}")

        return charts

def analyze_video_expressions(expressions_data: List[Tuple[int, str, float]], output_dir: Optional[str] = None, detector_backend: str = "default") -> Dict[str, str]:
    """
    Analyze expression data from a video and generate charts.

    Args:
        expressions_data: List of (frame_number, expression, confidence) tuples
        output_dir: Directory to save output charts
        detector_backend: Name of the detector backend used

    Returns:
        Dict[str, str]: Dictionary mapping chart types to file paths
    """
    # Create statistics analyzer
    stats = ExpressionStatistics(output_dir, detector_backend)

    # Add data
    for frame_num, expression, confidence in expressions_data:
        stats.add_frame_data([(expression, confidence)])

    # Generate all charts
    return stats.generate_all_charts()

def main():
    """
    Main function to demonstrate the statistics module.
    This can be run independently to test the module.
    """
    import random

    # Generate some random expression data for testing
    expressions_data = []
    for frame in range(1000):
        # Randomly select an expression with higher probability for some expressions
        # Exclude Neutral from the weights
        expressions = [e for e in config.EXPRESSIONS if e != "Neutral"]
        weights = [0.15, 0.1, 0.15, 0.3, 0.2, 0.1]  # Weights for each expression
        expression = random.choices(expressions, weights=weights)[0]

        # Generate a random confidence value
        confidence = random.uniform(0.5, 1.0)

        expressions_data.append((frame, expression, confidence))

    # Analyze the data and generate charts for each detector backend
    for backend in ['yolov8', 'yolov11n', 'yolov11s', 'yolov11m']:
        charts = analyze_video_expressions(expressions_data, detector_backend=backend)

        print(f"Generated charts for {backend}:")
        for chart_type, path in charts.items():
            if isinstance(path, list):
                print(f"{chart_type}: {len(path)} charts")
            else:
                print(f"{chart_type}: {path}")

if __name__ == "__main__":
    main()

