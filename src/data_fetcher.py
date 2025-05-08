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
            
        except Exception as e:
            logger.error(f"Error fetching data for {index_name}: {e}")
            
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    return sector_data

def fetch_stock_data(symbols):
    """
    Fetch historical stock data for the given symbols.
    
    Args:
        symbols (list): List of stock symbols
        
    Returns:
        dict: Dictionary mapping symbols to their historical data dataframes
    """
    end_date = datetime.now(INDIAN_TIMEZONE)
    start_date = end_date - timedelta(days=DATA_LOOKBACK_DAYS)
    
    stock_data = {}
    failed_symbols = []
    
    for symbol in tqdm(symbols, desc="Fetching stock data"):
        try:
            # Skip excluded symbols
            if symbol.replace(".NS", "") in EXCLUDED_SYMBOLS:
                continue
                
            # Yahoo Finance fetch for daily data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                failed_symbols.append(symbol)
                continue
                
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
            logger.error(f"Error fetching data for {symbol}: {e}")
            failed_symbols.append(symbol)
            
        # Add a small delay to avoid rate limiting
        time.sleep(0.2)
    
    if failed_symbols:
        logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols")
        
    logger.info(f"Successfully fetched data for {len(stock_data)} stocks")
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
