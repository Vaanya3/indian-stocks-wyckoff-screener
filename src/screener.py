"""
Main screener module for identifying Stage 1 to Stage 2 breakouts.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import yfinance as yf
from bs4 import BeautifulSoup
import requests

from .data_fetcher import DataCache  # Import the DataCache class
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
    INDIAN_TIMEZONE,
    NIFTY_500_SYMBOLS,
    CUSTOM_SYMBOLS,
    DATA_LOOKBACK_DAYS,
    WEEKLY_LOOKBACK_PERIODS,
    SECTOR_INDICES,
    EXCLUDED_SYMBOLS
)

logger = logging.getLogger(__name__)

def get_nifty_500_symbols_yf():
    """Fetches the list of Nifty 500 symbols from Wikipedia using BeautifulSoup."""
    nifty_500_url = "https://en.wikipedia.org/wiki/NIFTY_500"
    try:
        response = requests.get(nifty_500_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'lxml')
        tables = soup.find_all('table')
        nifty_500_table = None
        for table in tables:
            if 'constituents' in str(table.get('class')):
                nifty_500_table = table
                break
            elif 'wikitable' in str(table.get('class')) and 'sortable' in str(table.get('class')):
                header_row = table.find('tr')
                if header_row and 'Symbol' in header_row.text:
                    nifty_500_table = table
                    break

        if nifty_500_table:
            symbols = []
            for row in nifty_500_table.find_all('tr')[1:]:  # Skip the header row
                columns = row.find_all('td')
                if columns:
                    symbol = columns[0].text.strip()
                    symbol = symbol.replace(':', '').upper() + ".NS"
                    if symbol not in EXCLUDED_SYMBOLS:
                        symbols.append(symbol)
            return symbols
        else:
            logger.error("Could not find the Nifty 500 constituents table on Wikipedia using multiple checks.")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Nifty 500 symbols from Wikipedia: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing Nifty 500 symbols from Wikipedia: {e}")
        return []

def fetch_historical_data(symbol, period="1y", interval="1d"):
    """Fetches historical data for a given symbol using yfinance and caches it."""
    today = datetime.now()
    start_date = today - timedelta(days=DATA_LOOKBACK_DAYS)
    end_date = today

    cached_data = DataCache.get_stock_data(symbol, interval, start_date, end_date)
    if cached_data is not None:
        logger.info(f"Using cached {interval} data for {symbol}")
        return cached_data

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index = df.index.tz_localize(None) # Remove timezone info for consistency
            DataCache.save_stock_data(symbol, interval, df, start_date, end_date)
            logger.info(f"Fetched {interval} data for {symbol} from yfinance")
            return df
        else:
            logger.warning(f"No {interval} data found for {symbol} on yfinance.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching {interval} data for {symbol} from yfinance: {e}")
        return pd.DataFrame()

def fetch_sector_index_data(sector_index, period="1y"):
    """Fetches historical data for a sector index using yfinance and caches it."""
    today = datetime.now()
    start_date = today - timedelta(days=DATA_LOOKBACK_DAYS)
    end_date = today

    cached_data = DataCache.get_sector_data(sector_index)
    if cached_data is not None:
        logger.info(f"Using cached data for {sector_index}")
        return cached_data

    try:
        ticker = yf.Ticker(f"^{sector_index}.NS") # Using ^ for Yahoo Finance index tickers
        df = ticker.history(period=period, interval="1wk") # Using weekly for sector performance
        if not df.empty:
            df = df[['Close']]
            df.index = df.index.tz_localize(None)
            DataCache.save_sector_data(sector_index, df)
            logger.info(f"Fetched data for {sector_index} from yfinance")
            return df
        else:
            logger.warning(f"No data found for {sector_index} on yfinance.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data for {sector_index} from yfinance: {e}")
        return pd.DataFrame()

def fetch_all_data():
    """Fetches stock data, sector data, and creates the sector mapping."""
    logger.info("Fetching all required data...")
    all_symbols = []

    if NIFTY_500_SYMBOLS:
        nifty_500_symbols = get_nifty_500_symbols_yf()
        all_symbols.extend(nifty_500_symbols)

    all_symbols.extend(CUSTOM_SYMBOLS)
    all_symbols = list(set(all_symbols)) # Remove duplicates
    all_symbols = [s for s in all_symbols if s not in EXCLUDED_SYMBOLS]

    stock_data = {}
    for symbol in tqdm(all_symbols, desc="Fetching stock data"):
        daily_data = fetch_historical_data(symbol, period=f"{DATA_LOOKBACK_DAYS}d", interval="1d")
        weekly_data = fetch_historical_data(symbol, period=f"{WEEKLY_LOOKBACK_PERIODS * 7}d", interval="1wk") # Adjust period for weekly
        if not daily_data.empty and not weekly_data.empty:
            stock_data[symbol] = {'daily': daily_data, 'weekly': weekly_data}

    sector_data = {}
    for sector_index_name_config, sector_name_mapped in tqdm(SECTOR_INDICES.items(), desc="Fetching sector data"):
        sector_df = fetch_sector_index_data(sector_index_name_config, period=f"{WEEKLY_LOOKBACK_PERIODS * 7}d")
        if not sector_df.empty:
            sector_data[sector_index_name_config] = sector_df

    sector_mapping_func = lambda symbol: None
    # **IMPORTANT: Implement your accurate symbol-to-sector mapping here.**
    # The following is a basic example and likely needs to be replaced.
    symbol_to_sector = {}
    for symbol in stock_data.keys():
        found_sector = False
        for index_name, sector_name in SECTOR_INDICES.items():
            if any(part.lower() in symbol.lower() for part in index_name.split()):
                symbol_to_sector[symbol] = sector_name
                found_sector = True
                break
        if not found_sector:
            logger.warning(f"Could not determine sector for {symbol}")

    sector_mapping_func = symbol_to_sector.get

    logger.info(f"Fetched data for {len(stock_data)} stocks and {len(sector_data)} sectors.")
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

            if sector is None:
                logger.warning(f"Skipping {symbol} as its sector could not be determined.")
                continue

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
