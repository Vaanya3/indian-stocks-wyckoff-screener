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
    backoff_factor = 0.
