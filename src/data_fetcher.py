"""
Enhanced Data Fetching Module for Indian stock market data with improved error handling,
caching, and fallback mechanisms for NSE data access issues.
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
import hashlib
from requests.exceptions import ConnectionError, ReadTimeout
import aiohttp
import asyncio
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Callable
import sqlite3

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configuration (assumed to be in a config.py file)
try:
    from .config import (
        DATA_LOOKBACK_DAYS,
        NIFTY_500_SYMBOLS,
        CUSTOM_SYMBOLS,
        INDIAN_TIMEZONE,
        SECTOR_INDICES,
        EXCLUDED_SYMBOLS,
        CACHE_DIR,
        DB_PATH
    )
except ImportError:
    # Default configuration if import fails
    logger.warning("Config import failed. Using default values.")
    DATA_LOOKBACK_DAYS = 365
    NIFTY_500_SYMBOLS = True
    CUSTOM_SYMBOLS = []
    INDIAN_TIMEZONE = datetime.now().astimezone().tzinfo
    SECTOR_INDICES = {
        "NIFTY AUTO": "Auto",
        "NIFTY BANK": "Banking",
        "NIFTY FMCG": "FMCG",
        "NIFTY IT": "IT",
        "NIFTY METAL": "Metal",
        "NIFTY PHARMA": "Pharma",
        "NIFTY REALTY": "Realty",
    }
    EXCLUDED_SYMBOLS = []
    CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
    DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Better browser mimicking with rotating user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

def get_random_headers():
    """Generate random headers to avoid detection"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'DNT': '1',  # Do Not Track
    }

# Initialize SQLite database for persistent caching
def init_db():
    """Initialize SQLite database for persistent caching"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS symbol_lists (
        list_name TEXT PRIMARY KEY,
        symbols TEXT,
        timestamp TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        symbol TEXT,
        data_type TEXT,
        start_date TEXT,
        end_date TEXT,
        data BLOB,
        timestamp TIMESTAMP,
        PRIMARY KEY (symbol, data_type, start_date, end_date)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sector_data (
        sector_name TEXT PRIMARY KEY,
        data BLOB,
        timestamp TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Enhanced caching with SQLite backend
class DataCache:
    """Enhanced caching system for stock data"""
    
    @staticmethod
    def get_symbol_list(list_name: str, max_age_days: int = 7) -> Optional[List[str]]:
        """Get cached symbol list if it exists and is not too old"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT symbols, timestamp FROM symbol_lists WHERE list_name = ?", 
            (list_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            symbols_json, timestamp_str = result
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            if age.days < max_age_days:
                try:
                    return json.loads(symbols_json)
                except json.JSONDecodeError:
                    logger.error(f"Error decoding cached symbols for {list_name}")
        
        return None
    
    @staticmethod
    def save_symbol_list(list_name: str, symbols: List[str]) -> None:
        """Save symbol list to cache"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            symbols_json = json.dumps(symbols)
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT OR REPLACE INTO symbol_lists (list_name, symbols, timestamp) VALUES (?, ?, ?)",
                (list_name, symbols_json, timestamp)
            )
            conn.commit()
            logger.info(f"Cached {len(symbols)} symbols for {list_name}")
        except Exception as e:
            logger.error(f"Error caching symbols for {list_name}: {e}")
        finally:
            conn.close()
    
    @staticmethod
    def get_stock_data(symbol: str, data_type: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get cached stock data if available"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        cursor.execute(
            "SELECT data, timestamp FROM stock_data WHERE symbol = ? AND data_type = ? AND start_date = ? AND end_date = ?",
            (symbol, data_type, start_str, end_str)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data_blob, timestamp_str = result
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            # Max age for cache: 1 day for daily data, 7 days for weekly data
            max_age = 1 if data_type == 'daily' else 7
            
            if age.days < max_age:
                try:
                    return pd.read_json(data_blob)
                except Exception as e:
                    logger.error(f"Error decoding cached data for {symbol}: {e}")
        
        return None
    
    @staticmethod
    def save_stock_data(symbol: str, data_type: str, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> None:
        """Save stock data to cache"""
        if df is None or df.empty:
            return
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            data_json = df.to_json()
            timestamp = datetime.now().isoformat()
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            cursor.execute(
                """INSERT OR REPLACE INTO stock_data 
                   (symbol, data_type, start_date, end_date, data, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (symbol, data_type, start_str, end_str, data_json, timestamp)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
        finally:
            conn.close()
    
    @staticmethod
    def get_sector_data(sector_name: str, max_age_days: int = 7) -> Optional[pd.DataFrame]:
        """Get cached sector data if available"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT data, timestamp FROM sector_data WHERE sector_name = ?",
            (sector_name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data_blob, timestamp_str = result
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            if age.days < max_age_days:
                try:
                    return pd.read_json(data_blob)
                except Exception as e:
                    logger.error(f"Error decoding cached data for {sector_name}: {e}")
        
        return None
    
    @staticmethod
    def save_sector_data(sector_name: str, df: pd.DataFrame) -> None:
        """Save sector data to cache"""
        if df is None or df.empty:
            return
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            data_json = df.to_json()
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT OR REPLACE INTO sector_data (sector_name, data, timestamp) VALUES (?, ?, ?)",
                (sector_name, data_json, timestamp)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error caching data for {sector_name}: {e}")
        finally:
            conn.close()

# Fallback data for sector indices
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

# Fallback top NSE symbols for Nifty 500
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

async def fetch_with_retry(url: str, headers: Dict = None, timeout: int = 10, 
                          max_retries: int = 3, session: aiohttp.ClientSession = None) -> Optional[str]:
    """
    Fetch URL content with retry logic, using aiohttp for async requests
    """
    if headers is None:
        headers = get_random_headers()
    
    # Use provided session or create a new one
    if session is None:
        async with aiohttp.ClientSession() as session:
            return await _fetch_with_retry_impl(url, headers, timeout, max_retries, session)
    else:
        return await _fetch_with_retry_impl(url, headers, timeout, max_retries, session)

async def _fetch_with_retry_impl(url: str, headers: Dict, timeout: int, 
                               max_retries: int, session: aiohttp.ClientSession) -> Optional[str]:
    """Implementation of fetch with retry logic"""
    backoff_factor = 0.5
    attempts = 0
    
    while attempts < max_retries:
        try:
            # Add jitter to prevent synchronized retries
            jitter = random.uniform(0, 0.1)
            wait_time = (backoff_factor * (2 ** attempts)) + jitter
            
            if attempts > 0:
                logger.info(f"Retry attempt {attempts}/{max_retries} for {url} after {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                # Rotate headers to avoid detection patterns
                headers = get_random_headers()
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:  # Too Many Requests
                    logger.warning(f"Rate limited (429) when accessing {url}")
                    # Wait longer for rate limiting
                    await asyncio.sleep(5 + random.uniform(0, 5))
                else:
                    logger.warning(f"HTTP error {response.status} when accessing {url}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout accessing {url}")
        except Exception as e:
            logger.warning(f"Error accessing {url}: {e}")
            
        attempts += 1
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

async def get_nifty500_symbols_async() -> List[str]:
    """
    Fetch the list of Nifty 500 index components with enhanced async fetching.
    
    Returns:
        list: List of Nifty 500 stock symbols with NSE suffix
    """
    # Check cache first
    cached_symbols = DataCache.get_symbol_list("nifty500")
    if cached_symbols:
        logger.info(f"Using cached Nifty 500 symbols ({len(cached_symbols)} stocks)")
        return cached_symbols
    
    # Sources to try in order
    sources = [
        # Direct from NSE
        {
            "url": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
            "parser": lambda text: pd.read_csv(pd.StringIO(text)).Symbol.tolist()
        },
        # Alternative URL
        {
            "url": "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
            "parser": lambda text: pd.read_csv(pd.StringIO(text)).Symbol.tolist()
        },
        # Could add more alternative sources here
    ]
    
    async with aiohttp.ClientSession() as session:
        for source in sources:
            try:
                headers = {**get_random_headers(), 'Referer': 'https://www.nseindia.com/'}
                content = await fetch_with_retry(source["url"], headers=headers, session=session)
                
                if content:
                    symbols = source["parser"](content)
                    # Add .NS suffix for Yahoo Finance
                    symbols = [f"{symbol}.NS" for symbol in symbols]
                    
                    # Cache the successful result
                    DataCache.save_symbol_list("nifty500", symbols)
                    
                    logger.info(f"Successfully fetched {len(symbols)} Nifty 500 symbols")
                    return symbols
                    
            except Exception as e:
                logger.warning(f"Error fetching Nifty 500 from {source['url']}: {e}")
                continue
    
    # If we reach here, all sources failed - use fallback list
    logger.error("All Nifty 500 sources failed, using fallback symbols")
    DataCache.save_symbol_list("nifty500", NIFTY_FALLBACK_SYMBOLS)
    return NIFTY_FALLBACK_SYMBOLS

def get_nifty500_symbols() -> List[str]:
    """Synchronous wrapper for get_nifty500_symbols_async"""
    return asyncio.run(get_nifty500_symbols_async())

async def fetch_sector_indices_data_async() -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all sector indices with enhanced async approach.
    
    Returns:
        dict: Dictionary with sector index data
    """
    end_date = datetime.now(INDIAN_TIMEZONE)
    start_date = end_date - timedelta(days=DATA_LOOKBACK_DAYS)
    
    sector_data = {}
    
    for index_name in tqdm(SECTOR_INDICES.keys(), desc="Fetching sector indices"):
        # Check cache first
        cached_data = DataCache.get_sector_data(index_name)
        if cached_data is not None and not cached_data.empty:
            logger.info(f"Using cached data for {index_name}")
            sector_data[index_name] = cached_data
            continue
            
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
                        await asyncio.sleep(wait_time)
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
                
                # Cache the successful result
                DataCache.save_sector_data(index_name, weekly_data)
                
                success = True
                break  # Success, exit retry loop
                
            except (ConnectionError, ReadTimeout) as e:
                # Network-related errors, worth retrying
                logger.warning(f"Attempt {attempt+1}/{max_retries} - Network error for {index_name}: {e}")
                if attempt < max_retries - 1:
                    # Add exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying after {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
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
                
                # Cache the fallback data
                DataCache.save_sector_data(index_name, weekly_data)
                
            except Exception as e:
                logger.error(f"Error using fallback data for {index_name}: {e}")
                
        # Add a small delay to avoid overwhelming the API
        await asyncio.sleep(0.5)
    
    return sector_data

def get_sector_indices_data() -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper for fetch_sector_indices_data_async"""
    return asyncio.run(fetch_sector_indices_data_async())

async def download_stock_data_async(symbol: str, start_date: datetime, end_date: datetime, 
                                   max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Download stock data with retries and alternative methods using async.
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        DataFrame or None: Stock data if successful, None otherwise
    """
    # Check cache first
    cached_data = DataCache.get_stock_data(symbol, 'daily', start_date, end_date)
    if cached_data is not None and not cached_data.empty:
        logger.debug(f"Using cached data for {symbol}")
        return cached_data
    
    # Helper function to download with yfinance (synchronous)
    def _download_yf(sym, start, end, timeout_val=10):
        try:
            return yf.download(
                sym,
                start=start,
                end=end,
                progress=False,
                show_errors=False,
                timeout=timeout_val
            )
        except Exception as e:
            logger.debug(f"Error in yfinance download for {sym}: {e}")
            return pd.DataFrame()
    
    # First try yfinance's download with exponential backoff
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Add exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            
            # We use an executor to run the synchronous yfinance code
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, 
                lambda: _download_yf(symbol, start_date, end_date)
            )
            
            if not df.empty:
                # Cache the successful result
                DataCache.save_stock_data(symbol, 'daily', df, start_date, end_date)
                return df
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} - Error downloading {symbol}: {e}")
    
    # If download failed, try Ticker history
    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        
        df = await loop.run_in_executor(
            None,
            lambda: ticker.history(period=f"{DATA_LOOKBACK_DAYS}d")
        )
        
        if not df.empty:
            # Cache the successful result
            DataCache.save_stock_data(symbol, 'daily', df, start_date, end_date)
            return df
            
    except Exception as e:
        logger.warning(f"Ticker history also failed for {symbol}: {e}")
    
    # If both methods failed, try alternative symbol
    if '.NS' in symbol:
        alt_symbol = symbol.replace('.NS', '')
        logger.info(f"Trying alternative symbol format: {alt_symbol}")
        
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: _download_yf(alt_symbol, start_date, end_date)
            )
            
            if not df.empty:
                # Cache the successful result
                DataCache.save_stock_data(symbol, 'daily', df, start_date, end_date)
                return df
                
        except Exception as e:
            logger.warning(f"Alternative symbol {alt_symbol} also failed: {e}")
    
    # If we get here, try the BSE symbol as a last resort
    try:
        bse_symbol = symbol.replace('.NS', '.BO')
        logger.info(f"Trying BSE symbol: {bse_symbol}")
        
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: _download_yf(bse_symbol, start_date, end_date)
        )
        
        if not df.empty:
            # Cache the successful result
            DataCache.save_stock_data(symbol, 'daily', df, start_date, end_date)
            return df
            
    except Exception as e:
        logger.warning(f"BSE symbol {bse_symbol} also failed: {e}")
    
    return None

def download_stock_data(symbol: str, start_date: datetime, end_date: datetime, 
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Synchronous wrapper for download_stock_data_async"""
    return asyncio.run(download_stock_data_async(symbol, start_date, end_date, max_retries))

async async def fetch_stock_data_async(symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch historical stock data for the given symbols using async methods.
    
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
    batch_size = 10  # Reduced batch size for better reliability
    symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(symbol_batches, desc="Processing symbol batches")):
        # Add a longer delay between batches if not the first batch
        if batch_idx > 0:
            await asyncio.sleep(2 + random.uniform(0, 1))
        
        # Process this batch
        batch_results = await process_symbol_batch_async(batch, start_date, end_date)
        stock_data.update(batch_results)
    
    # Count failures
    failed_count = len(symbols) - len(stock_data)
    if failed_count > 0:
        logger.warning(f"Failed to fetch data for {failed_count}/{len(symbols)} symbols")
        
    logger.info(f"Successfully fetched data for {len(stock_data)}/{len(symbols)} stocks")
    return stock_data

def fetch_stock_data(symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Synchronous wrapper for fetch_stock_data_async"""
    return asyncio.run(fetch_stock_data_async(symbols))

async def process_symbol_batch_async(batch: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Process a batch of symbols asynchronously"""
    batch_data = {}
    tasks = []
    
    # Create a semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(5)  # Allow 5 concurrent downloads
    
    async def download_with_semaphore(symbol):
        async with semaphore:
            return symbol, await download_stock_data_async(symbol, start_date, end_date)
    
    # Create download tasks for all symbols in batch
    for symbol in batch:
        # Skip excluded symbols
        if symbol.replace(".NS", "") in EXCLUDED_SYMBOLS:
            continue
        
        tasks.append(download_with_semaphore(symbol))
    
    # Wait for all downloads to complete
    results = await asyncio.gather(*tasks)
    
    # Process results
    for symbol, df in results:
        if df is None or df.empty:
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
            
            # Save weekly data to cache
            DataCache.save_stock_data(symbol, 'weekly', weekly_data, start_date, end_date)
            
            # Store both daily and weekly data
            batch_data[symbol] = {
                'daily': df.reset_index(),
                'weekly': weekly_data
            }
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
    
    return batch_data
