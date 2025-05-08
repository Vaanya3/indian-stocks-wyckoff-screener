"""
Utility functions for the Indian Stocks Wyckoff Screener.
"""
import os
import json
import pandas as pd
import logging
from datetime import datetime
from .config import INDIAN_TIMEZONE, CHART_SAVE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    Create necessary directories for the application.
    """
    os.makedirs(CHART_SAVE_DIR, exist_ok=True)
    logger.info(f"Directory structure set up at {CHART_SAVE_DIR}")

def save_results(results, filename=None):
    """
    Save screening results to a JSON file.
    
    Args:
        results (list): Screening results
        filename (str, optional): Custom filename
        
    Returns:
        str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now(INDIAN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        filename = f"wyckoff_breakouts_{timestamp}.json"
    
    results_path = os.path.join(os.path.dirname(CHART_SAVE_DIR), "results", filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Convert DataFrame columns to JSON serializable format
    serializable_results = []
    for result in results:
        result_copy = result.copy()
        
        # Remove DataFrame objects
        for key in list(result_copy.keys()):
            if isinstance(result_copy[key], pd.DataFrame):
                del result_copy[key]
        
        serializable_results.append(result_copy)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results_path

def load_results(filename):
    """
    Load screening results from a JSON file.
    
    Args:
        filename (str): Results filename
        
    Returns:
        list: Screening results
    """
    results_path = os.path.join(os.path.dirname(CHART_SAVE_DIR), "results", filename)
    
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return []
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded results from {results_path}")
    return results

def clean_old_charts(days=7):
    """
    Clean up old chart files.
    
    Args:
        days (int): Remove charts older than this many days
    """
    import time
    from datetime import timedelta
    
    now = time.time()
    cutoff = now - (days * 24 * 60 * 60)
    
    count = 0
    for filename in os.listdir(CHART_SAVE_DIR):
        file_path = os.path.join(CHART_SAVE_DIR, filename)
        if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
            os.remove(file_path)
            count += 1
    
    logger.info(f"Cleaned up {count} chart files older than {days} days")

def format_summary(results):
    """
    Format screening results as a readable summary.
    
    Args:
        results (list): Screening results
        
    Returns:
        str: Formatted summary
    """
    if not results:
        return "No breakouts detected."
    
    summary = f"📊 WYCKOFF BREAKOUT SUMMARY 📊\n"
    summary += f"Scan Date: {datetime.now(INDIAN_TIMEZONE).strftime('%Y-%m-%d %H:%M')}\n"
    summary += f"Stocks Found: {len(results)}\n\n"
    
    # Group by sector
    sectors = {}
    for result in results:
        sector = result.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(result)
    
    # Display by sector
    for sector, stocks in sorted(sectors.items()):
        summary += f"\n🔹 {sector} ({len(stocks)} stocks)\n"
        
        for stock in stocks:
            symbol = stock['symbol'].replace('.NS', '')
            breakout_percent = ((stock['current_price'] / stock['resistance_level']) - 1) * 100
            summary += f"  • {symbol}: ₹{stock['current_price']:.2f} (+{breakout_percent:.1f}% above resistance)\n"
    
    return summary
