"""
Configuration settings for the Indian Stocks Wyckoff Screener.
"""
import os
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Data settings
INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')
DATA_LOOKBACK_DAYS = 365  # Days of historical data to fetch
WEEKLY_LOOKBACK_PERIODS = 52  # Number of weekly periods to analyze

# Stock universe
NIFTY_500_SYMBOLS = True  # Use Nifty 500 index components
CUSTOM_SYMBOLS = []  # Add any additional symbols here

# Technical parameters
# Moving Averages
MA_SHORT = 10  # 10-week moving average
MA_LONG = 30   # 30-week moving average

# Base identification
MIN_BASE_WEEKS = 12  # Minimum 3 months of sideways action
MAX_BASE_VOLATILITY = 0.15  # Maximum volatility in the base
MIN_FLAT_MA_SLOPE = -0.02  # Maximum negative slope for "flattening" MA
MIN_RISING_MA_SLOPE = 0.02  # Minimum positive slope for "rising" MA

# Breakout criteria
VOLUME_INCREASE_THRESHOLD = 0.5  # 50% above 20-day average
RESISTANCE_LOOKBACK = 90  # Days to look back for resistance levels
BREAKOUT_THRESHOLD = 0.02  # 2% above resistance confirms breakout

# Momentum filters
SECTOR_OUTPERFORMANCE = 0.05  # 5% outperformance vs sector
MFI_PERIOD = 14  # Money Flow Index period
MFI_THRESHOLD = 50  # MFI threshold for accumulation

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Sector analysis
SECTOR_RANKING_PERIOD = 13  # 13-week performance for sector ranking
TOP_SECTOR_PERCENTILE = 0.5  # Top 50% of sectors

# Chart settings
CHART_TYPE = 'candle'  # 'candle', 'line', 'renko'
CHART_STYLE = 'classic'  # 'classic', 'yahoo', 'charles', 'binance', 'ibd'
CHART_LOOKBACK_PERIODS = 52  # Weeks to show in chart
VOLUME_PANEL_HEIGHT = 0.2  # Height ratio for volume panel
MACD_PANEL_HEIGHT = 0.2  # Height ratio for MACD panel
MFI_PANEL_HEIGHT = 0.2  # Height ratio for MFI panel

# File paths
CHART_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'charts')
os.makedirs(CHART_SAVE_DIR, exist_ok=True)

# Indian market indices for sector mapping
SECTOR_INDICES = {
    'NIFTY AUTO': 'Automobile',
    'NIFTY BANK': 'Banking',
    'NIFTY FINANCIAL SERVICES': 'Financial Services',
    'NIFTY FMCG': 'Consumer Goods',
    'NIFTY IT': 'Information Technology',
    'NIFTY MEDIA': 'Media',
    'NIFTY METAL': 'Metal',
    'NIFTY PHARMA': 'Pharmaceutical',
    'NIFTY PSU BANK': 'Public Sector Banks',
    'NIFTY REALTY': 'Real Estate',
    'NIFTY ENERGY': 'Energy',
    'NIFTY COMMODITIES': 'Commodities',
    'NIFTY CONSUMPTION': 'Consumption',
    'NIFTY INFRA': 'Infrastructure',
    'NIFTY MNC': 'Multinational Companies',
    'NIFTY PSE': 'Public Sector Enterprises',
    'NIFTY HEALTHCARE': 'Healthcare',
}

# List of symbols to exclude (optional)
EXCLUDED_SYMBOLS = []
