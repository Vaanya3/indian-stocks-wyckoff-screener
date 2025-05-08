"""
Data fetching module for Indian stock market data.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from nsepy import get_history
from nsepy import get_index_pe_history
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
import random
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

def get_nifty500_symbols():
    """
    Fetch the list of Nifty 500 index components.
    
    Returns:
        list: List of Nifty 500 stock symbols with NSE suffix
    """
    try:
        # Try to fetch from NSE website
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df = pd.read_csv(url)
        symbols = df['Symbol'].tolist()
        # Add .NS suffix for Yahoo Finance
        symbols = [f"{symbol}.NS" for symbol in symbols]
        logger.info(f"Successfully fetched {len(symbols)} Nifty 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching Nifty 500 symbols: {e}")
        # Fallback to a local file or a smaller default list
        logger.info("Using default top 100 symbols as fallback")
        # You can maintain a local backup of major symbols
        top_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "ITC.NS", "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS",
            "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS"
        ]
        return top_symbols

def get_sector_indices_data():
    """
    Fetch data for all sector indices.
    
    Returns:
        dict: Dictionary with sector index data
    """
    end_date = datetime.now(INDIAN_TIMEZONE)
    start_date = end_date - timedelta(days=DATA_LOOKBACK_DAYS)
    
    sector_data = {}
    
    for index_name in tqdm(SECTOR_INDICES.keys(), desc="Fetching sector indices"):
        max_retries = 3
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
                    continue
                    
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
                
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    return sector_data

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
    batch_size = 50
    symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
    
    for batch in tqdm(symbol_batches, desc="Processing symbol batches"):
        for symbol in tqdm(batch, desc="Fetching stock data", leave=False):
            # Skip excluded symbols
            if symbol.replace(".NS", "") in EXCLUDED_SYMBOLS:
                continue
                
            # Try with retries and different symbol formats
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        # Add exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                    
                    # Yahoo Finance fetch for daily data
                    ticker = yf.Ticker(symbol)
                    
                    # Use download instead of history for better reliability
                    df = yf.download(
                        symbol, 
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False,
                        timeout=10
                    )
                    
                    # If empty and not the last attempt, continue to next attempt
                    if df.empty and attempt < max_retries - 1:
                        continue
                    
                    # If empty on final attempt, mark as failed
                    if df.empty:
                        logger.warning(f"{symbol}: Returned empty dataset")
                        failed_symbols.append(symbol)
                        break
                    
                    # Validate data structure
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_columns):
                        logger.warning(f"{symbol}: Missing required columns")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            failed_symbols.append(symbol)
                            break
                    
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
                    
                    success = True
                    break  # Successfully fetched data, exit retry loop
                    
                except (ConnectionError, ReadTimeout) as e:
                    # Network errors, worth retrying
                    logger.warning(f"Attempt {attempt+1}/{max_retries} - Network error for {symbol}: {e}")
                
                except Exception as e:
                    # Log the error
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    
                    # If we're on the last attempt and still failed
                    if attempt == max_retries - 1:
                        failed_symbols.append(symbol)
            
            # If the normal symbol failed, try an alternative format
            if not success and '.NS' in symbol:
                alt_symbol = symbol.replace('.NS', '')
                logger.info(f"Trying alternative symbol format: {alt_symbol}")
                
                try:
                    # Try with the alternative symbol
                    ticker = yf.Ticker(alt_symbol)
                    df = yf.download(
                        alt_symbol, 
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False
                    )
                    
                    if not df.empty:
                        # Add symbol column - use the original symbol for consistency
                        df['Symbol'] = symbol  # Use original symbol for consistency in the data
                        
                        # Compute weekly data
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
                        
                        # Remove from failed symbols if it was there
                        if symbol in failed_symbols:
                            failed_symbols.remove(symbol)
                            
                        logger.info(f"Successfully fetched {alt_symbol} as alternative for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Alternative symbol {alt_symbol} also failed: {e}")
            
            # Add a small random delay to avoid rate limiting
            time.sleep(0.2 + random.uniform(0, 0.3))
        
        # Add a larger delay between batches
        time.sleep(1 + random.uniform(0, 1))
    
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
