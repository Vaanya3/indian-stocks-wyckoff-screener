"""
Telegram alerts module for sending stock breakout notifications.
"""
import os
import logging
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import time
from .config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    CHART_SAVE_DIR,
    MA_SHORT,
    MA_LONG,
    CHART_LOOKBACK_PERIODS,
    CHART_TYPE,
    CHART_STYLE,
    VOLUME_PANEL_HEIGHT,
    MACD_PANEL_HEIGHT,
    MFI_PANEL_HEIGHT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_chart(daily_df, weekly_df, symbol, resistance_level=None):
    """
    Generate a chart for the stock that shows the breakout.
    
    Args:
        daily_df (pd.DataFrame): Daily price data
        weekly_df (pd.DataFrame): Weekly price data
        symbol (str): Stock symbol
        resistance_level (float, optional): Resistance level to mark
        
    Returns:
        str: Path to saved chart
    """
    # Prepare weekly data for plotting
    plot_df = weekly_df.set_index(pd.DatetimeIndex(weekly_df['Date']))
    
    # Limit data to lookback period
    plot_df = plot_df.tail(CHART_LOOKBACK_PERIODS)
    
    # Create subplots
    fig, axes = mpf.plot(
        plot_df,
        type=CHART_TYPE,
        style=CHART_STYLE,
        title=f'\n{symbol} - Stage 1 to Stage 2 Breakout',
        volume=True,
        panel_ratios=(1, VOLUME_PANEL_HEIGHT, MACD_PANEL_HEIGHT, MFI_PANEL_HEIGHT),
        figsize=(12, 10),
        addplot=[
            # Moving averages
            mpf.make_addplot(plot_df[f'MA_{MA_SHORT}'], color='blue', width=1),
            mpf.make_addplot(plot_df[f'MA_{MA_LONG}'], color='red', width=1),
            # MACD in subplot
            mpf.make_addplot(plot_df['MACD'], panel=2, color='fuchsia', secondary_y=False),
            mpf.make_addplot(plot_df['MACD_Signal'], panel=2, color='blue', secondary_y=False),
            # MFI in subplot
            mpf.make_addplot(plot_df['MFI'], panel=3, color='green', secondary_y=False),
        ],
        returnfig=True
    )
    
    # Add resistance level line if provided
    if resistance_level:
        axes[0].axhline(y=resistance_level, color='r', linestyle='--', alpha=0.7)
        axes[0].text(0, resistance_level, f'  Resistance: {resistance_level:.2f}', 
                     color='r', fontsize=9, va='bottom')
    
    # Add MFI threshold line
    axes[3].axhline(y=50, color='k', linestyle='--', alpha=0.5)
    
    # Label the panels
    axes[2].set_ylabel('MACD')
    axes[3].set_ylabel('MFI')
    
    # Save chart
    chart_filename = f"{symbol.replace('.NS', '')}_breakout_{int(time.time())}.png"
    chart_path = os.path.join(CHART_SAVE_DIR, chart_filename)
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close(fig)
    
    return chart_path

def send_telegram_message(message_text):
    """
    Send a text message to Telegram.
    
    Args:
        message_text (str): Message to send
        
    Returns:
        bool: Success status
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info(f"Message sent to Telegram: {message_text[:50]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_chart(chart_path, caption):
    """
    Send a chart image to Telegram.
    
    Args:
        chart_path (str): Path to the chart image
        caption (str): Caption for the chart
        
    Returns:
        bool: Success status
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    try:
        with open(chart_path, 'rb') as photo:
            files = {
                'photo': photo
            }
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            
            logger.info(f"Chart sent to Telegram: {chart_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to send Telegram chart: {e}")
        return False

def send_breakout_alert(symbol, stock_data, sector, resistance_level, daily_close):
    """
    Send a complete breakout alert to Telegram.
    
    Args:
        symbol (str): Stock symbol
        stock_data (dict): Stock data dictionary
        sector (str): Stock sector
        resistance_level (float): Identified resistance level
        daily_close (float): Latest daily closing price
        
    Returns:
        bool: Success status
    """
    try:
        # Generate chart
        chart_path = generate_chart(
            stock_data['daily'], 
            stock_data['weekly'], 
            symbol, 
            resistance_level
        )
        
        # Create alert message
        breakout_percent = ((daily_close / resistance_level) - 1) * 100
        
        message = f"🚀 <b>STAGE 2 BREAKOUT DETECTED</b> 🚀\n\n"
        message += f"<b>Stock:</b> {symbol.replace('.NS', '')}\n"
        message += f"<b>Sector:</b> {sector}\n"
        message += f"<b>Breakout Level:</b> ₹{resistance_level:.2f}\n"
        message += f"<b>Current Price:</b> ₹{daily_close:.2f} (+{breakout_percent:.2f}%)\n"
        message += f"<b>10-Week MA:</b> ₹{stock_data['weekly'][f'MA_{MA_SHORT}'].iloc[-1]:.2f}\n"
        message += f"<b>30-Week MA:</b> ₹{stock_data['weekly'][f'MA_{MA_LONG}'].iloc[-1]:.2f}\n"
        message += f"<b>Volume Surge:</b> {stock_data['daily']['Volume_Ratio'].iloc[-1]:.1f}x avg\n"
        message += f"<b>MFI:</b> {stock_data['daily']['MFI'].iloc[-1]:.1f}\n\n"
        message += "This stock has completed Stage 1 accumulation and is now entering Stage 2 markup phase."
        
        # Send chart with caption
        return send_telegram_chart(chart_path, message)
    
    except Exception as e:
        logger.error(f"Error sending breakout alert for {symbol}: {e}")
        return False
