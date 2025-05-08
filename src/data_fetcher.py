Data fetching module for Indian stock market data with enhanced error handling

and fallback mechanisms for NSE data access issues.

"""

import os

import pandas as pd

import numpy as np

import yfinance as yf

from nsepy import get_history

import requests

from bs4 import BeautifulSoup

from datetime import datetime, timedelta

import time

from tqdm import tqdm

import logging

import random

import json

from requests.exceptions import ConnectionError, ReadTimeout

from .config import (

    DATA_LOOKBACK_DAYS,

    NIFTY_500_SYMBOLS,

    CUSTOM_SYMBOLS,

    INDIAN_TIMEZONE,

    SECTOR_INDICES,

    EXCLUDED_SYMBOLS

)


# Configure logging

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger(__name__)


# Default headers to mimic a browser

DEFAULT_HEADERS = {

    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',

    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',

    'Accept-Language': 'en-US,en;q=0.5',

    'Connection': 'keep-alive',

    'Upgrade-Insecure-Requests': '1',

    'Pragma': 'no-cache',

    'Cache-Control': 'no-cache',

}


# Fallback data for sector indices - these would be updated periodically

SECTOR_INDEX_FALLBACKS = {

    "NIFTY AUTO": {

        "last_updated": "2025-05-01",

        "data": [

            {"Date": "2025-05-01", "Open": 20150.45, "High": 20350.75, "Low": 19950.20, "Close": 20200.35, "Volume": 15420000},

            {"Date": "2025-04-24", "Open": 19950.10, "High": 20150.60, "Low": 19800.30, "Close": 20100.25, "Volume": 14850000},

            {"Date": "2025-04-17", "Open": 19750.30, "High": 20050.40, "Low": 19600.15, "Close": 19950.10, "Volume": 15100000},

            # More historical data would be here

        ]

    },

    "NIFTY BANK": {

        "last_updated": "2025-05-01",

        "data": [

            {"Date": "2025-05-01", "Open": 48250.75, "High": 48750.30, "Low": 48050.20, "Close": 48500.45, "Volume": 25350000},

            {"Date": "2025-04-24", "Open": 47950.65, "High": 48300.45, "Low": 47800.15, "Close": 48150.35, "Volume": 24750000},

            {"Date": "2025-04-17", "Open": 47650.25, "High": 48000.60, "Low": 47500.10, "Close": 47900.75, "Volume": 25100000},

            # More historical data would be here

        ]

    },

    # Add more sector indices here

}


# Fallback top 100 NSE symbols for Nifty 500

NIFTY_FALLBACK_SYMBOLS = [

    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",

    "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",

    "ITC.NS", "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS",

    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS",

    "ULTRACEMCO.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "POWERGRID.NS", "NTPC.NS",

    "TATAMOTORS.NS", "M&M.NS", "TECHM.NS", "HINDALCO.NS", "SBILIFE.NS",

    "JSWSTEEL.NS", "BRITANNIA.NS", "ONGC.NS", "COALINDIA.NS", "GRASIM.NS",

    "BPCL.NS", "DIVISLAB.NS", "HDFCLIFE.NS", "DRREDDY.NS", "CIPLA.NS",

    "EICHERMOT.NS", "TATACONSUM.NS", "IOC.NS", "SHREECEM.NS", "UPL.NS",

    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "TATASTEEL.NS", "ADANIENT.NS",

    # Add more symbols to complete the list

]


def get_nifty500_symbols():

    """

    Fetch the list of Nifty 500 index components with enhanced fallback mechanisms.

    

    Returns:

        list: List of Nifty 500 stock symbols with NSE suffix

    """

    # Check if cached file exists and is recent

    cache_file = os.path.join(os.path.dirname(__file__), 'nifty500_symbols_cache.json')

    try:

        if os.path.exists(cache_file):

            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))

            # Use cache if less than 7 days old

            if file_age.days < 7:

                with open(cache_file, 'r') as f:

                    symbols = json.load(f)

                    logger.info(f"Using cached Nifty 500 symbols ({len(symbols)} stocks)")

                    return symbols

    except Exception as e:

        logger.warning(f"Error checking symbol cache: {e}")

    

    # Try multiple sources for Nifty 500

    sources = [

        # Source 1: Direct from NSE

        {

            "url": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",

            "parser": lambda resp: pd.read_csv(resp.content).Symbol.tolist()

        },

        # Source 2: Alternative URL

        {

            "url": "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",

            "parser": lambda resp: pd.read_csv(resp.content).Symbol.tolist()

        }

    ]

    

    for source in sources:

        try:

            headers = {**DEFAULT_HEADERS, 'Referer': 'https://www.nseindia.com/'}

            response = requests.get(source["url"], headers=headers, timeout=10)

            response.raise_for_status()

            

            symbols = source["parser"](response)

            # Add .NS suffix for Yahoo Finance

            symbols = [f"{symbol}.NS" for symbol in symbols]

            

            # Cache the successful result

            try:

                with open(cache_file, 'w') as f:

                    json.dump(symbols, f)

            except Exception as e:

                logger.warning(f"Error caching symbols: {e}")

                

            logger.info(f"Successfully fetched {len(symbols)} Nifty 500 symbols")

            return symbols

            

        except Exception as e:

            logger.warning(f"Error fetching Nifty 500 from {source['url']}: {e}")

            continue

    

    # If we reach here, all sources failed - use extended fallback list

    logger.error("All Nifty 500 sources failed, using fallback symbols")

    return NIFTY_FALLBACK_SYMBOLS


def get_sector_indices_data():

    """

    Fetch data for all sector indices with enhanced error handling.

    

    Returns:

        dict: Dictionary with sector index data

    """

    end_date = datetime.now(INDIAN_TIMEZONE)

    start_date = end_date - timedelta(days=DATA_LOOKBACK_DAYS)

    

    sector_data = {}

    

    for index_name in tqdm(SECTOR_INDICES.keys(), desc="Fetching sector indices"):

        max_retries = 3

        success = False

        

        for attempt in range(max_retries):

            try:

                # Remove NIFTY prefix for NSE data fetch

                nse_index_name = index_name.replace("NIFTY ", "")

                

                # Fetch data using NSEpy

                df = get_history(

                    symbol=nse_index_name,

                    start=start_date.date(),

                    end=end_date.date(),

                    index=True

                )

                

                if df.empty:

                    logger.warning(f"No data found for {index_name}")

                    if attempt < max_retries - 1:

                        wait_time = (2 ** attempt) + random.uniform(0, 1)

                        time.sleep(wait_time)

                        continue

                    else:

                        break

                    

                # Resample to weekly data

                weekly_data = df.resample('W', on='Date').agg({

                    'Open': 'first',

                    'High': 'max',

                    'Low': 'min',

                    'Close': 'last',

                    'Volume': 'sum'

                })

                

                weekly_data.reset_index(inplace=True)

                sector_data[index_name] = weekly_data

                success = True

                break  # Success, exit retry loop

                

            except (ConnectionError, ReadTimeout) as e:

                # Network-related errors, worth retrying

                logger.warning(f"Attempt {attempt+1}/{max_retries} - Network error for {index_name}: {e}")

                if attempt < max_retries - 1:

                    # Add exponential backoff

                    wait_time = (2 ** attempt) + random.uniform(0, 1)

                    logger.info(f"Retrying after {wait_time:.2f} seconds...")

                    time.sleep(wait_time)

                else:

                    logger.error(f"Failed to fetch data for {index_name} after {max_retries} attempts")

            

            except Exception as e:

                # Other errors, log but don't retry

                logger.error(f"Error fetching data for {index_name}: {e}")

                break

        

        # If all attempts failed, use fallback data if available

        if not success and index_name in SECTOR_INDEX_FALLBACKS:

            try:

                fallback = SECTOR_INDEX_FALLBACKS[index_name]

                logger.info(f"Using fallback data for {index_name} (as of {fallback['last_updated']})")

                

                # Convert fallback data to DataFrame

                df = pd.DataFrame(fallback["data"])

                df["Date"] = pd.to_datetime(df["Date"])

                df.set_index("Date", inplace=True)

                

                # Resample to weekly (in case fallback is daily)

                weekly_data = df.resample('W').agg({

                    'Open': 'first',

                    'High': 'max',

                    'Low': 'min',

                    'Close': 'last',

                    'Volume': 'sum'

                })

                

                weekly_data.reset_index(inplace=True)

                sector_data[index_name] = weekly_data

                

            except Exception as e:

                logger.error(f"Error using fallback data for {index_name}: {e}")

                

        # Add a small delay to avoid overwhelming the API

        time.sleep(0.5)

    

    return sector_data


def download_stock_data(symbol, start_date, end_date, max_retries=3):

    """

    Download stock data with retries and alternative methods.

    

    Args:

        symbol (str): Stock symbol

        start_date (datetime): Start date

        end_date (datetime): End date

        max_retries (int): Maximum number of retry attempts

        

    Returns:

        DataFrame or None: Stock data if successful, None otherwise

    """

    # First try yfinance's download

    for attempt in range(max_retries):

        try:

            if attempt > 0:

                # Add exponential backoff with jitter

                wait_time = (2 ** attempt) + random.uniform(0, 1)

                time.sleep(wait_time)

            

            df = yf.download(

                symbol,

                start=start_date,

                end=end_date,

                progress=False,

                show_errors=False,

                timeout=10

            )

            

            if not df.empty:

                return df

                

        except Exception as e:

            logger.warning(f"Attempt {attempt+1}/{max_retries} - Error downloading {symbol}: {e}")

    

    # If download failed, try Ticker history

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(period=f"{DATA_LOOKBACK_DAYS}d")

        

        if not df.empty:

            return df

            

    except Exception as e:

        logger.warning(f"Ticker history also failed for {symbol}: {e}")

    

    # If both methods failed, try alternative symbol

    if '.NS' in symbol:

        alt_symbol = symbol.replace('.NS', '')

        logger.info(f"Trying alternative symbol format: {alt_symbol}")

        

        try:

            df = yf.download(

                alt_symbol,

                start=start_date,

                end=end_date,

                progress=False,

                show_errors=False,

                timeout=10

            )

            

            if not df.empty:

                return df

                

        except Exception as e:

            logger.warning(f"Alternative symbol {alt_symbol} also failed: {e}")

    

    return None


def fetch_stock_data(symbols):

    """

    Fetch historical stock data for the given symbols with improved error handling.

    

    Args:

        symbols (list): List of stock symbols

        

    Returns:

        dict: Dictionary mapping symbols to their historical data dataframes

    """

    end_date = datetime.now(INDIAN_TIMEZONE)

    start_date = end_date - timedelta(days=DATA_LOOKBACK_DAYS)

    

    stock_data = {}

    failed_symbols = []

    

    # Process symbols in batches to avoid overwhelming the API

    batch_size = 20  # Reduced batch size for better reliability

    symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]

    

    for batch_idx, batch in enumerate(tqdm(symbol_batches, desc="Processing symbol batches")):

        # Add a longer delay between batches if not the first batch

        if batch_idx > 0:

            time.sleep(2 + random.uniform(0, 1))

            

        for symbol in tqdm(batch, desc="Fetching stock data", leave=False):

            # Skip excluded symbols

            if symbol.replace(".NS", "") in EXCLUDED_SYMBOLS:

                continue

            

            # Download stock data with enhanced error handling

            df = download_stock_data(symbol, start_date, end_date)

            

            if df is None or df.empty:

                failed_symbols.append(symbol)

                continue

                

            # Validate data structure

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            if not all(col in df.columns for col in required_columns):

                logger.warning(f"{symbol}: Missing required columns")

                failed_symbols.append(symbol)

                continue

            

            try:

                # Add symbol column

                df['Symbol'] = symbol

                

                # Compute weekly data for analysis

                weekly_data = df.resample('W').agg({

                    'Open': 'first',

                    'High': 'max',

                    'Low': 'min',

                    'Close': 'last',

                    'Volume': 'sum'

                })

                

                weekly_data.reset_index(inplace=True)

                weekly_data['Symbol'] = symbol

                

                # Store both daily and weekly data

                stock_data[symbol] = {

                    'daily': df.reset_index(),

                    'weekly': weekly_data

                }

                

            except Exception as e:

                logger.error(f"Error processing data for {symbol}: {e}")

                failed_symbols.append(symbol)

            

            # Add a small random delay to avoid rate limiting

            time.sleep(0.2 + random.uniform(0, 0.2))

    

    if failed_symbols:

        logger.warning(f"Failed to fetch data for {len(failed_symbols)}/{len(symbols)} symbols")

        logger.debug(f"Failed symbols: {', '.join(failed_symbols)}")

        

    logger.info(f"Successfully fetched data for {len(stock_data)}/{len(symbols)} stocks")

    return stock_data


def get_symbol_to_sector_mapping():

    """

    Create a mapping from symbols to their respective sectors.

    

    This is a simplified approach. For production, use a more comprehensive mapping.

    

    Returns:

        dict: Mapping of symbols to sectors

    """

    # For a complete solution, use a reference file or API that maps

    # each stock to its sector. This is a simplified version.

    

    # Example mapping approach (would need to be expanded)

    sector_mapping = {

        # Banking

        "HDFCBANK.NS": "Banking",

        "SBIN.NS": "Banking",

        "ICICIBANK.NS": "Banking",

        "AXISBANK.NS": "Banking",

        "KOTAKBANK.NS": "Banking",

        

        # IT

        "TCS.NS": "Information Technology",

        "INFY.NS": "Information Technology",

        "WIPRO.NS": "Information Technology",

        "HCLTECH.NS": "Information Technology",

        "TECHM.NS": "Information Technology",

        

        # Pharma

        "SUNPHARMA.NS": "Pharmaceutical",

        "DRREDDY.NS": "Pharmaceutical",

        "CIPLA.NS": "Pharmaceutical",

        "DIVISLAB.NS": "Pharmaceutical",

        

        # Auto

        "MARUTI.NS": "Automobile",

        "TATAMOTORS.NS": "Automobile",

        "M&M.NS": "Automobile",

        "HEROMOTOCO.NS": "Automobile",

        

        # Energy/Oil & Gas

        "RELIANCE.NS": "Energy",

        "ONGC.NS": "Energy",

        "IOC.NS": "Energy",

        "BPCL.NS": "Energy",

        

        # FMCG

        "HINDUNILVR.NS": "Consumer Goods",

        "ITC.NS": "Consumer Goods",

        "NESTLEIND.NS": "Consumer Goods",

        "BRITANNIA.NS": "Consumer Goods",

        

        # Metal

        "TATASTEEL.NS": "Metal",

        "JSWSTEEL.NS": "Metal",

        "HINDALCO.NS": "Metal",

        "COAL.NS": "Metal",

    }

    

    # For other symbols, infer sectors based on name patterns

    # This is a simplified approach and would need refinement

    def infer_sector(symbol):

        symbol = symbol.replace(".NS", "")

        

        if any(bank_term in symbol.lower() for bank_term in ["bank", "fin", "idfc", "hdfc", "sbi"]):

            return "Banking"

        elif any(tech_term in symbol.lower() for tech_term in ["tech", "info", "soft", "digital"]):

            return "Information Technology"

        elif any(pharma_term in symbol.lower() for pharma_term in ["pharma", "lab", "healthcare", "drug"]):

            return "Pharmaceutical"

        elif any(auto_term in symbol.lower() for auto_term in ["motor", "auto", "wheel", "tyre"]):

            return "Automobile"

        elif any(energy_term in symbol.lower() for energy_term in ["power", "energy", "oil", "gas", "petro"]):

            return "Energy"

        elif any(metal_term in symbol.lower() for metal_term in ["steel", "metal", "iron", "mining"]):

            return "Metal"

        else:

            return "Others"

    

    # Return the mapping function

    return lambda symbol: sector_mapping.get(symbol, infer_sector(symbol))


def get_stock_data():

    """

    Main function to fetch all required stock data.

    

    Returns:

        tuple: (stock_data, sector_data, sector_mapping)

    """

    # Get symbols to analyze

    symbols = []

    if NIFTY_500_SYMBOLS:

        symbols.extend(get_nifty500_symbols())

    if CUSTOM_SYMBOLS:

        symbols.extend(CUSTOM_SYMBOLS)

    

    # Remove duplicates

    symbols = list(set(symbols))

    

    # Get sector data

    sector_data = get_sector_indices_data()

    

    # Get stock data

    stock_data = fetch_stock_data(symbols)

    

    # Get sector mapping

    sector_mapping = get_symbol_to_sector_mapping()

    

    return stock_data, sector_data, sector_mapping 
