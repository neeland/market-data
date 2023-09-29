import yfinance as yf
import pandas_datareader as pdr
from alpha_vantage.timeseries import TimeSeries
import finnhub
#import talib
#import zipline as zp
import pyfolio as pf
import pandas as pd
import matplotlib.pyplot as plt
    
import matplotlib.pyplot as plt
import seaborn as sns

def get_symbol_data(symbol: str, start_date: str, end_date: str, interval: str='1d', source_dict: dict = None) -> dict:
    """
    Pull symbol data from various Python packages.

    Args:
    symbol (str): Stock symbol to download
    start_date (str): Start date of data in YYYY-MM-DD format 
    end_date (str): End date of data in YYYY-MM-DD format
    interval (str): Data interval, either '1d' or '1mo'
    source_dict (dict): Optional dictionary containing package sources for each symbol

    Returns:
    dict: Dictionary containing DataFrames from each package
    """
    if source_dict is None:
        source_dict = {'yfinance': symbol,}
                       #'alphavantage': symbol,
                       #'pandas_datareader': symbol,
                       #'finnhub': symbol}

    yf_data = yf.download(source_dict['yfinance'], start=start_date, end=end_date, interval=interval)
    
    #av_data = TimeSeries(key='B92FX7PM8WHTQVUV').get_daily(source_dict['alphavantage'], outputsize='full')[0]
    
    #pdr_data = pdr.get_data_yahoo(source_dict['pandas_datareader'], start=start_date, end=end_date, interval=interval)
    
    #finnhub_client = finnhub.Client(api_key="YOUR_FINNHUB_API_KEY") 
    #finnhub_data = finnhub_client.stock_candles(source_dict['finnhub'], 'D', start_date, end_date)
    
    #talib_data = talib.SMA(yf_data['Close'], timeperiod=20)

    # Zipline
    #prices = {source_dict['zipline']: yf_data['Close']}
    #benchmark_returns, strategy_returns = zp.run_algorithm(start=start_date, end=end_date, initialize=initialize,
    #                                                        handle_data=handle_data, capital_base=10000, 
    #                                                        bundle='quandl')

    # Pyfolio
    #returns = pd.Series(strategy_returns)
    #pf.create_returns_tear_sheet(returns)

    return {'yfinance': yf_data,}
            #'alphavantage': av_data,}
            #'pandas_datareader': pdr_data,
            #'finnhub': finnhub_data,
            #'talib': talib_data,
            #'zipline': benchmark_returns,
            #'pyfolio': returns}


def get_symbols_data(symbols: list, start_date: str, end_date: str, interval: str='1d', source_dict: dict = None) -> dict:
    """
    Pull data for multiple symbols from various Python packages.

    Args:
    symbols (list): List of stock symbols to download
    start_date (str): Start date of data in YYYY-MM-DD format 
    end_date (str): End date of data in YYYY-MM-DD format
    interval (str): Data interval, either '1d' or '1mo'
    source_dict (dict): Optional dictionary containing package sources for each symbol

    Returns:
    dict: Dictionary containing DataFrames from each package for each symbol
    """
    symbol_data = {}
    for symbol in symbols:
        symbol_data[symbol] = get_symbol_data(symbol, start_date, end_date, interval, source_dict)
    return symbol_data

if __name__ == "__main__":


    # setting ticker list of Apple, Meta, Amazon, Netflix, Google, Microsoft, Tesla, Nvidia, AMD, Oracle, Intel, Space X, as a list
    ticker_list = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'NVDA', 'AMD', 'ORCL', 'INTC', 'SPCE']
    start_date = '2000-01-01'
    end_date = '2023-09-01'

    data = get_symbols_data(ticker_list, start_date, end_date)

    # list of sns styles
    sns_styles = ["darkgrid", "whitegrid", "dark", "white", "ticks" ]
    
    # set style
    sns.set_style("darkgrid")

    # plot data
    fig, ax = plt.subplots(figsize=(12, 8))
    for symbol in data:
        # plot
        data[symbol]['yfinance']['Close'].plot(ax=ax, label=symbol)
        # add label
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f"Closing price from {start_date} to {end_date}", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('stock.png')    
    print("fin.")