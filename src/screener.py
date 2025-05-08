"""
Main screener module for identifying Stage 1 to Stage 2 breakouts.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from .data_fetcher import DataCache  # Import the DataCache class
# from .data_fetcher import get_stock_data # Remove the direct import
from .analysis import (
    add_technical_indicators,
    identify_resistance_level,
    is_in_base,
    is_breaking_out,
    check_momentum_filters,
    rank_sectors,
    is_sector_in_stage2,
    get_sector_performance
)
from .telegram_alerts import send_breakout_alert
from .utils import setup_directories, save_results, format_summary, clean_old_charts
from .config import (
    TOP_SECTOR_PERCENTILE,
    INDIAN_TIMEZONE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_all_data():
    """
    Placeholder function to fetch all required data (stock data, sector data, mapping).
    You need to implement the actual logic here using DataCache or other methods.
    """
    logger.warning("fetch_all_data in screener.py is a placeholder. Implement your data fetching logic.")
    # Example of how you might use DataCache (you'll need to adapt this):
    # symbol = "RELIANCE.NS"
    # today = datetime.now()
    # start_date = today - timedelta(days=365)
    # daily_data = DataCache.get_stock_data(symbol, 'daily', start_date, today)
    # ... similarly for other data

    # Replace this with your actual data fetching and mapping logic
    stock_data = {}
    sector_data = {}
    sector_mapping_func = lambda symbol: None
    return stock_data, sector_data, sector_mapping_func

def run_screen():
    """
    Run the Wyckoff screening process.

    Returns:
        list: List of breakout stocks with details
    """
    logger.info("Starting Wyckoff Stage 1 to Stage 2 breakout screening")

    # Setup directories
    setup_directories()

    # Clean old chart files
    clean_old_charts()

    # Get stock data
    logger.info("Fetching stock data...")
    stock_data, sector_data, sector_mapping = fetch_all_data()
    logger.info(f"Data fetched for {len(stock_data)} stocks and {len(sector_data)} sectors")

    # Rank sectors
    logger.info("Ranking sectors...")
    sector_ranks = rank_sectors(sector_data)

    # Track results
    breakout_stocks = []

    # Process each stock
    logger.info("Screening stocks for Wyckoff Stage 1 to Stage 2 breakouts...")
    for symbol, data in tqdm(stock_data.items(), desc="Analyzing stocks"):
        try:
            # Get sector for this stock
            sector = sector_mapping(symbol)

            # Skip if sector is not in top percentile
            sector_indices = [s for s in sector_ranks.keys() if sector in sector_data[s]]
            if sector_indices and sector_ranks.get(sector_indices[0], 1) > TOP_SECTOR_PERCENTILE:
                continue

            # Skip if sector is not in Stage 2
            if not any(is_sector_in_stage2(sector_data, idx) for idx in sector_indices):
                continue

            # Add technical indicators to data
            daily_df = add_technical_indicators(data['daily'])
            weekly_df = add_technical_indicators(data['weekly'])

            # Step 1: Check if stock was in a Stage 1 base
            if not is_in_base(weekly_df):
                continue

            # Step 2: Identify resistance level
            resistance_level = identify_resistance_level(daily_df)

            # Step 3: Check if breaking out
            if not is_breaking_out(daily_df, weekly_df, resistance_level):
                continue

            # Step 4: Apply momentum filters
            sector_perf = get_sector_performance(sector_data, sector)
            if not check_momentum_filters(daily_df, weekly_df, sector_perf):
                continue

            # This stock is breaking out from Stage 1 to Stage 2!
            current_price = daily_df['Close'].iloc[-1]
            logger.info(f"Breakout detected: {symbol}, Price: {current_price}, Resistance: {resistance_level}")

            # Record the breakout
            breakout_info = {
                'symbol': symbol,
                'sector': sector,
                'scan_date': datetime.now(INDIAN_TIMEZONE).strftime('%Y-%m-%d'),
                'resistance_level': float(resistance_level),
                'current_price': float(current_price),
                'volume_ratio': float(daily_df['Volume_Ratio'].iloc[-1]),
                'mfi': float(daily_df['MFI'].iloc[-1]),
                'ma10': float(weekly_df[f'MA_10'].iloc[-1]),
                'ma30': float(weekly_df[f'MA_30'].iloc[-1])
            }

            breakout_stocks.append(breakout_info)

            # Send alert to Telegram
            send_breakout_alert(
                symbol,
                {'daily': daily_df, 'weekly': weekly_df},
                sector,
                resistance_level,
                current_price
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    # Save results
    if breakout_stocks:
        save_results(breakout_stocks)

        # Print summary
        summary = format_summary(breakout_stocks)
        print("\n" + summary)

    logger.info(f"Screening complete. Found {len(breakout_stocks)} breakout stocks.")
    return breakout_stocks

if __name__ == "__main__":
    run_screen()
