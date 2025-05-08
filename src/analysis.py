"""
Technical analysis functions for identifying Wyckoff Stage 1 to Stage 2 transitions.
"""
import pandas as pd
import numpy as np
import ta
from .config import (
    MA_SHORT, 
    MA_LONG, 
    MIN_BASE_WEEKS, 
    MAX_BASE_VOLATILITY,
    MIN_FLAT_MA_SLOPE,
    MIN_RISING_MA_SLOPE,
    VOLUME_INCREASE_THRESHOLD,
    RESISTANCE_LOOKBACK,
    BREAKOUT_THRESHOLD,
    SECTOR_OUTPERFORMANCE,
    MFI_PERIOD,
    MFI_THRESHOLD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL
)

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df (pd.DataFrame): Stock price dataframe
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Add moving averages
    df[f'MA_{MA_SHORT}'] = df['Close'].rolling(window=MA_SHORT).mean()
    df[f'MA_{MA_LONG}'] = df['Close'].rolling(window=MA_LONG).mean()
    
    # Add slopes of moving averages (for detecting flat or rising MA)
    df[f'MA_{MA_LONG}_Slope'] = df[f'MA_{MA_LONG}'].diff(periods=4) / df[f'MA_{MA_LONG}'].shift(periods=4)
    
    # Add volatility measure (standard deviation of percentage changes)
    df['Volatility'] = df['Close'].pct_change().rolling(window=MIN_BASE_WEEKS).std()
    
    # Add volume indicators
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Add Money Flow Index
    df['MFI'] = ta.volume.money_flow_index(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=MFI_PERIOD,
        fillna=True
    )
    
    # Add MACD
    macd = ta.trend.MACD(
        close=df['Close'],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL
    )
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    return df

def identify_resistance_level(df, lookback=RESISTANCE_LOOKBACK):
    """
    Identify the resistance level for a stock.
    
    Args:
        df (pd.DataFrame): Stock price dataframe
        lookback (int): Number of periods to look back
        
    Returns:
        float: Identified resistance level
    """
    # Use recent high as a simple resistance level
    # A more sophisticated approach would use clustering of highs
    recent_df = df.tail(lookback)
    
    # Find clusters of highs
    highs = recent_df['High'].values
    
    # Simple approach: use the most recent significant high
    # For more sophisticated approach, we could use clustering algorithms
    resistance_level = np.percentile(highs, 85)  # 85th percentile of highs
    
    return resistance_level

def is_in_base(weekly_df):
    """
    Check if a stock is in a Stage 1 base (accumulation).
    
    Args:
        weekly_df (pd.DataFrame): Weekly stock data
        
    Returns:
        bool: True if in Stage 1 base, False otherwise
    """
    # Ensure we have enough data
    if len(weekly_df) < MIN_BASE_WEEKS:
        return False
    
    recent_df = weekly_df.tail(MIN_BASE_WEEKS)
    
    # Check for sideways price action (low volatility)
    if recent_df['Volatility'].mean() > MAX_BASE_VOLATILITY:
        return False
    
    # Check for flattening 30-week MA
    ma_slope = recent_df[f'MA_{MA_LONG}_Slope'].tail(4).mean()
    if ma_slope < MIN_FLAT_MA_SLOPE:
        return False  # Still in downtrend, not flattened yet
    
    # Price should be relatively stable (trading in a range)
    price_range = (recent_df['High'].max() - recent_df['Low'].min()) / recent_df['Low'].min()
    if price_range > 0.25:  # More than 25% range is not a tight base
        return False
    
    return True

def is_breaking_out(daily_df, weekly_df, resistance_level):
    """
    Check if a stock is breaking out from a Stage 1 base.
    
    Args:
        daily_df (pd.DataFrame): Daily stock data
        weekly_df (pd.DataFrame): Weekly stock data
        resistance_level (float): Identified resistance level
        
    Returns:
        bool: True if breaking out, False otherwise
    """
    # Get the most recent data
    last_daily = daily_df.iloc[-1]
    
    # Check if price is breaking above resistance
    if last_daily['Close'] < resistance_level * (1 + BREAKOUT_THRESHOLD):
        return False
    
    # Check for increased volume
    if last_daily['Volume_Ratio'] < (1 + VOLUME_INCREASE_THRESHOLD):
        return False
    
    # Check for golden cross (10-week MA crossing above 30-week MA)
    if weekly_df[f'MA_{MA_SHORT}'].iloc[-1] <= weekly_df[f'MA_{MA_LONG}'].iloc[-1]:
        return False
    
    # Check for rising 30-week MA
    ma_slope = weekly_df[f'MA_{MA_LONG}_Slope'].tail(4).mean()
    if ma_slope < MIN_RISING_MA_SLOPE:
        return False
    
    return True

def check_momentum_filters(daily_df, weekly_df, sector_performance=0):
    """
    Apply momentum filters to confirm Stage 1 to Stage 2 transition.
    
    Args:
        daily_df (pd.DataFrame): Daily stock data
        weekly_df (pd.DataFrame): Weekly stock data
        sector_performance (float): Sector performance as benchmark
        
    Returns:
        bool: True if passes momentum filters, False otherwise
    """
    # Check MFI is above threshold and rising
    recent_mfi = daily_df['MFI'].tail(5)
    if recent_mfi.iloc[-1] < MFI_THRESHOLD or recent_mfi.iloc[-1] < recent_mfi.iloc[-5]:
        return False
    
    # Check for positive MACD or recent crossover
    if daily_df['MACD'].iloc[-1] < 0 and daily_df['MACD_Hist'].iloc[-1] < 0:
        # No positive MACD or recent bullish crossover
        return False
    
    # Calculate 1-month performance
    one_month_ago = daily_df.iloc[-21]['Close'] if len(daily_df) > 21 else daily_df.iloc[0]['Close']
    stock_performance = (daily_df.iloc[-1]['Close'] / one_month_ago) - 1
    
    # Compare to sector performance
    if stock_performance < (sector_performance + SECTOR_OUTPERFORMANCE):
        return False  # Not outperforming sector by required threshold
    
    return True

def get_sector_performance(sector_data, sector_name, weeks=4):
    """
    Calculate sector performance over the specified period.
    
    Args:
        sector_data (dict): Dictionary of sector index dataframes
        sector_name (str): Name of the sector
        weeks (int): Number of weeks to calculate performance
        
    Returns:
        float: Sector performance as decimal (e.g., 0.05 for 5%)
    """
    # Find matching sector index
    matching_sectors = [s for s in sector_data.keys() if sector_name in sector_data[s]]
    
    if not matching_sectors:
        return 0  # Default if no matching sector
    
    sector_df = sector_data[matching_sectors[0]]
    
    if len(sector_df) <= weeks:
        return 0
    
    # Calculate performance
    current_price = sector_df['Close'].iloc[-1]
    past_price = sector_df['Close'].iloc[-1 - weeks]
    
    return (current_price / past_price) - 1

def rank_sectors(sector_data, weeks=13):
    """
    Rank sectors by their performance over the specified period.
    
    Args:
        sector_data (dict): Dictionary of sector index dataframes
        weeks (int): Number of weeks for ranking
        
    Returns:
        dict: Dictionary mapping sector names to performance rank percentile
    """
    performances = {}
    
    for sector_name, df in sector_data.items():
        if len(df) <= weeks:
            performances[sector_name] = 0
            continue
            
        current_price = df['Close'].iloc[-1]
        past_price = df['Close'].iloc[-1 - weeks]
        
        performance = (current_price / past_price) - 1
        performances[sector_name] = performance
    
    # Sort by performance
    sorted_sectors = sorted(performances.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate percentile ranks (0 = best, 1 = worst)
    n_sectors = len(sorted_sectors)
    ranks = {}
    
    for i, (sector, _) in enumerate(sorted_sectors):
        ranks[sector] = i / n_sectors
    
    return ranks

def is_sector_in_stage2(sector_data, sector_name):
    """
    Determine if a sector is in Stage 2 (markup phase).
    
    Args:
        sector_data (dict): Dictionary of sector index dataframes
        sector_name (str): Name of the sector
        
    Returns:
        bool: True if sector is in Stage 2, False otherwise
    """
    # Find matching sector index
    matching_sectors = [s for s in sector_data.keys() if sector_name in s]
    
    if not matching_sectors:
        return False  # Default if no matching sector
    
    sector_df = sector_data[matching_sectors[0]]
    
    # Add MAs if they don't exist
    if f'MA_{MA_SHORT}' not in sector_df.columns:
        sector_df[f'MA_{MA_SHORT}'] = sector_df['Close'].rolling(window=MA_SHORT).mean()
    
    if f'MA_{MA_LONG}' not in sector_df.columns:
        sector_df[f'MA_{MA_LONG}'] = sector_df['Close'].rolling(window=MA_LONG).mean()
    
    # Check if shorter MA is above longer MA
    ma_short_above = sector_df[f'MA_{MA_SHORT}'].iloc[-1] > sector_df[f'MA_{MA_LONG}'].iloc[-1]
    
    # Check if longer MA is rising
    ma_long_rising = sector_df[f'MA_{MA_LONG}'].iloc[-1] > sector_df[f'MA_{MA_LONG}'].iloc[-5]
    
    # Check if current price is above both MAs
    price_above_mas = (
        sector_df['Close'].iloc[-1] > sector_df[f'MA_{MA_SHORT}'].iloc[-1] and
        sector_df['Close'].iloc[-1] > sector_df[f'MA_{MA_LONG}'].iloc[-1]
    )
    
    # Stage 2 criteria: MAs aligned correctly and price above MAs
    return ma_short_above and ma_long_rising and price_above_mas
