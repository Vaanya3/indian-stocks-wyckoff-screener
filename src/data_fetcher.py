import os
import sys
import warnings
import logging
from time import sleep
from datetime import datetime, timedelta
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
import pandas as pd
import yfinance as yf
from yfinance import shared
from PKDevTools.classes.Utils import USER_AGENTS
import random
from yfinance.version import version as yfVersion
if yfVersion == "0.2.28":
    from yfinance.data import TickerData as YfData
    class YFPricesMissingError(Exception):
        pass
    class YFInvalidPeriodError(Exception):
        pass
    class YFRateLimitError(Exception):
        pass
else:
    from yfinance.data import YfData
    from yfinance.exceptions import YFPricesMissingError, YFInvalidPeriodError, YFRateLimitError
from concurrent.futures import ThreadPoolExecutor
from PKDevTools.classes.PKDateUtilities import PKDateUtilities
from PKDevTools.classes.ColorText import colorText
from PKDevTools.classes.Fetcher import StockDataEmptyException
from PKDevTools.classes.log import default_logger
from PKDevTools.classes.SuppressOutput import SuppressOutput
from PKNSETools.PKNSEStockDataFetcher import nseStockDataFetcher
from pkscreener.classes.PKTask import PKTask
from PKDevTools.classes.OutputControls import OutputControls
from PKDevTools.classes import Archiver

# Configure logging (if not already configured elsewhere)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

class screenerStockDataFetcher(nseStockDataFetcher):
    _tickersInfoDict = {}

    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.configManager = config_manager  # Ensure configManager is accessible

    def download_stock_data_with_retries(self, symbol, start_date, end_date, max_retries=3):
        """
        Download stock data with retries using yfinance.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    show_errors=False,
                    timeout=self.configManager.generalTimeout / 4 if self.configManager else 10
                )

                if not df.empty:
                    return df

            except Exception as e:
                default_logger().warning(f"Attempt {attempt+1}/{max_retries} - Error downloading {symbol}: {e}")
                time.sleep(1) # Small delay even on general exceptions

        return None

    def fetchStockDataWithArgs(self, *args):
        task = None
        if isinstance(args[0], PKTask):
            task = args[0]
            stockCode, period, duration, exchangeSuffix = task.long_running_fn_args
        else:
            stockCode, period, duration, exchangeSuffix = args[0], args[1], args[2], args[3]
        result = self.fetchStockData(stockCode, period, duration, None, 0, 0, 0, exchangeSuffix=exchangeSuffix, printCounter=False)
        if task is not None:
            if task.taskId >= 0:
                task.progressStatusDict[task.taskId] = {'progress': 0, 'total': 1}
                task.resultsDict[task.taskId] = result
                task.progressStatusDict[task.taskId] = {'progress': 1, 'total': 1}
            task.result = result
        return result

    def get_stats(self, ticker):
        info = yf.Tickers(ticker).tickers[ticker].fast_info
        screenerStockDataFetcher._tickersInfoDict[ticker] = {"marketCap": info.market_cap if info is not None else 0}

    def fetchAdditionalTickerInfo(self, ticker_list, exchangeSuffix=".NS"):
        if not isinstance(ticker_list, list):
            raise TypeError("ticker_list must be a list")
        if len(exchangeSuffix) > 0:
            ticker_list = [(f"{x}{exchangeSuffix}" if not x.endswith(exchangeSuffix) else x) for x in ticker_list]
        screenerStockDataFetcher._tickersInfoDict = {}
        with ThreadPoolExecutor() as executor:
            executor.map(self.get_stats, ticker_list)
        return screenerStockDataFetcher._tickersInfoDict

    def fetchStockData(
        self,
        stockCode,
        period,
        duration,
        proxyServer=None,
        screenResultsCounter=0,
        screenCounter=0,
        totalSymbols=0,
        printCounter=False,
        start=None,
        end=None,
        exchangeSuffix=".NS",
        attempt=0
    ):
        if isinstance(stockCode, list):
            stockCode = [(f"{x}{exchangeSuffix}" if (not x.endswith(exchangeSuffix) and not x.startswith("^")) else x) for x in stockCode]
        elif isinstance(stockCode, str):
            stockCode = f"{stockCode}{exchangeSuffix}" if (not stockCode.endswith(exchangeSuffix) and not stockCode.startswith("^")) else stockCode

        if (period in ["1d", "5d", "1mo", "3mo", "5mo"] or duration[-1] in ["m", "h"]):
            start = None
            end = None

        data = None
        with SuppressOutput(suppress_stdout=(not printCounter), suppress_stderr=(not printCounter)):
            if isinstance(stockCode, str):
                end_date = PKDateUtilities.currentDateTime().date()
                if period and period[-1] == 'y':
                    start_date = (end_date - timedelta(days=int(period[:-1]) * 365))
                elif period and period[-1] == 'm':
                    start_date = (end_date - timedelta(days=int(period[:-1]) * 30))
                elif period and period[-1] == 'd':
                    start_date = (end_date - timedelta(days=int(period[:-1])))
                else: # Default lookback of 1 year
                    start_date = (end_date - timedelta(days=365))

                data = self.download_stock_data_with_retries(stockCode, start_date, end_date)

            elif isinstance(stockCode, list):
                all_data = {}
                for ticker in stockCode:
                    end_date = PKDateUtilities.currentDateTime().date()
                    if period and period[-1] == 'y':
                        start_date = (end_date - timedelta(days=int(period[:-1]) * 365))
                    elif period and period[-1] == 'm':
                        start_date = (end_date - timedelta(days=int(period[:-1]) * 30))
                    elif period and period[-1] == 'd':
                        start_date = (end_date - timedelta(days=int(period[:-1])))
                    else: # Default lookback of 1 year
                        start_date = (end_date - timedelta(days=365))

                    df = self.download_stock_data_with_retries(ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        all_data[ticker] = df
                if all_data:
                    data = pd.concat(all_data, axis=1, keys=all_data.keys())
                else:
                    data = pd.DataFrame()

        if printCounter and type(screenCounter) != int:
            sys.stdout.write("\r\033[K")
            try:
                OutputControls().printOutput(
                    colorText.GREEN
                    + (
                        "[%d%%] Screened %d, Found %d. Fetching data & Analyzing %s..."
                        % (
                            int((screenCounter.value / totalSymbols) * 100),
                            screenCounter.value,
                            screenResultsCounter.value,
                            stockCode,
                        )
                    )
                    + colorText.END,
                    end="",
                )
            except ZeroDivisionError:
                pass
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                default_logger().debug(e, exc_info=True)
                pass

        if (data is None or len(data) == 0) and printCounter:
            OutputControls().printOutput(
                colorText.FAIL
                + "=> Failed to fetch!"
                + colorText.END,
                end="\r",
                flush=True,
            )
            raise StockDataEmptyException
        if printCounter:
            OutputControls().printOutput(
                colorText.GREEN + "=> Done!" + colorText.END,
                end="\r",
                flush=True,
            )
        return data

    def fetchLatestNiftyDaily(self, proxyServer=None):
        end_date = PKDateUtilities.currentDateTime().date()
        start_date = end_date - timedelta(days=5)
        data = self.download_stock_data_with_retries("^NSEI", start_date, end_date)
        return data

    def fetchFiveEmaData(self, proxyServer=None):
        end_date = PKDateUtilities.currentDateTime()
        start_date_5d = end_date - timedelta(days=5)

        nifty_sell = self.download_stock_data_with_retries("^NSEI", start_date_5d, end_date, max_retries=2)
        banknifty_sell = self.download_stock_data_with_retries("^NSEBANK", start_date_5d, end_date, max_retries=2)
        nifty_buy = self.download_stock_data_with_retries("^NSEI", start_date_5d, end_date, max_retries=2)
        banknifty_buy = self.download_stock_data_with_retries("^NSEBANK", start_date_5d, end_date, max_retries=2)

        return nifty_buy, banknifty_buy, nifty_sell, banknifty_sell

    def fetchWatchlist(self):
        createTemplate = False
        data = pd.DataFrame()
        try:
            data = pd.read_excel("watchlist.xlsx")
        except FileNotFoundError as e:
            default_logger().debug(e, exc_info=True)
            OutputControls().printOutput(
                colorText.FAIL
                + f"  [+] watchlist.xlsx not found in {os.getcwd()}"
                + colorText.END
            )
            createTemplate = True
        try:
            if not createTemplate:
                data = data["Stock Code"].values.tolist()
        except KeyError as e:
            default_logger().debug(e, exc_info=True)
            OutputControls().printOutput(
                colorText.FAIL
                + '  [+] Bad Watchlist Format: First Column (A1) should have Header named "Stock Code"'
                + colorText.END
            )
            createTemplate = True
        if createTemplate:
            sample = {"Stock Code": ["SBIN", "INFY", "TATAMOTORS", "ITC"]}
            sample_data = pd.DataFrame(sample, columns=["Stock Code"])
            sample_data.to_excel("watchlist_template.xlsx", index=False, header=True)
            OutputControls().printOutput(
                colorText.BLUE
                + f"  [+] watchlist_template.xlsx created in {os.getcwd()} as a referance template."
                + colorText.END
            )
            return None
        return data

if __name__ == '__main__':
    # Example usage (you might need to adapt this based on your overall application)
    from PKDevTools.classes.ConfigManager import ConfigManager
    config_manager = ConfigManager() # Initialize your ConfigManager
    fetcher = screenerStockDataFetcher(config_manager=config_manager)

    try:
        # Fetch data for a single stock
        stock_code = "RELIANCE.NS"
        period = "1y"
        duration = "1d"
        stock_data = fetcher.fetchStockData(stock_code, period, duration, printCounter=True)
        if stock_data is not None and not stock_data.empty:
            print(f"\nFetched data for {stock_code}:\n{stock_data.head()}")
        else:
            print(f"\nFailed to fetch data for {stock_code}")

        # Fetch data for multiple stocks
        stock_codes = ["TCS.NS", "INFY.NS", "HDFCBANK.NS"]
        period = "3mo"
        duration = "1wk"
        multiple_stock_data = fetcher.fetchStockData(stock_codes, period, duration, printCounter=True)
        if multiple_stock_data is not None and not multiple_stock_data.empty:
            print(f"\nFetched data for {stock_codes[0]}:\n{multiple_stock_data[stock_codes[0]].head()}")
            print(f"\nFetched data for {stock_codes[1]}:\n{multiple_stock_data[stock_codes[1]].head()}")
            print(f"\nFetched data for {stock_codes[2]}:\n{multiple_stock_data[stock_codes[2]].head()}")
        else:
            print(f"\nFailed to fetch data for {stock_codes}")

        # Fetch Nifty daily data
        nifty_data = fetcher.fetchLatestNiftyDaily()
        if nifty_data is not None and not nifty_data.empty:
            print(f"\nLatest Nifty Daily Data:\n{nifty_data.head()}")
        else:
            print("\nFailed to fetch Nifty daily data.")

        # Fetch watchlist
        watchlist = fetcher.fetchWatchlist()
        if watchlist:
            print(f"\nWatchlist: {watchlist}")
            if watchlist:
                watchlist_data = fetcher.fetchStockData(watchlist, "1mo", "1d", printCounter=True)
                if watchlist_data is not None and not watchlist_data.empty:
                    print("\nFetched data for watchlist.")
                else:
                    print("\nFailed to fetch data for watchlist.")
        else:
            print("\nCould not load watchlist.")

    except StockDataEmptyException:
        print("\nStockDataEmptyException occurred.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
