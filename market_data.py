import yfinance as yf
import pandas_datareader as pdr
#from alpha_vantage.timeseries import TimeSeries
import pickle
#import finnhub
#import talib
#import zipline as zp
#import pyfolio as pf

import matplotlib.pyplot as plt
import seaborn as sns

START_DATE = '2020-09-01'
END_DATE = '2023-09-29'

# setting ticker list of Apple, Meta, Amazon, Netflix, Google, Microsoft, Tesla, Nvidia, AMD, Oracle, Intel, Space X, IBM, Palintir  as a list
QUERY_TICKER_LIST = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'NVDA', 'AMD', 'ORCL', 'INTC', 'SPCE', 'IBM', 'PLTR'  ] 

def get_symbol_data(symbol: str, max_period: bool, start_date: str, end_date: str, interval: str='1d', source_dict: dict = None) -> dict:
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
    if max_period:
            yf_data = yf.download(source_dict['yfinance'], period="max", interval=interval)
    else:
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


def get_symbols_list_data(symbols: list, start_date: str, end_date: str, interval: str='1d', source_dict: dict = None) -> dict:
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

def sns_plots(data, plot_columns, plot_types, start_date, end_date, sns_style='darkgrid'):
    """
    Plots stock data for a given column and time range using Seaborn.

    Args:
        data (dict): A dictionary containing stock data for different symbols.
        start_date (str): Start date of the time range in the format 'YYYY-MM-DD'.
        end_date (str): End date of the time range in the format 'YYYY-MM-DD'.
        column (str): The column to plot. Must be one of ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
        sns_style (str, optional): The Seaborn style to use. Must be one of ["darkgrid", "whitegrid", "dark", "white", "ticks"]. Defaults to 'darkgrid'.

    Raises:
        ValueError: If the provided column or sns_style is invalid.

    Returns:
        None
    """
    # check if sns_style is valid
    # list of sns styles
    valid_sns_styles = ["darkgrid", "whitegrid", "dark", "white", "ticks" ]
    if sns_style not in valid_sns_styles:
        raise ValueError(f"Invalid column '{column}'. Column must be one of {valid_sns_styles}")
    sns.set_style(sns_style)
    
    # check if column is valid
    valid_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for column in plot_columns:
        if column not in valid_columns:
            raise ValueError(f"Invalid column '{column}'. Column must be one of {valid_columns}")

    # plot data
    fig, ax = plt.subplots(nrows=len(plot_columns), ncols=1, figsize=(12, 16))
    for symbol in data:
        for i, column in enumerate(plot_columns):
            print(f"Plotting {symbol} : {column}")
            # plot
            data[symbol]['yfinance'][column].plot(kind=plot_types[i], ax=ax[i], label=f"{symbol} : {column}", legend=True)      
            # add label
            plt.legend(loc='upper left', fontsize=16)
            # set axis title
            ax[i].set_title(f"{column} : {start_date} to {end_date}", fontsize=20)
            plt.xlabel("Date", fontsize=16)
            plt.ylabel("Price (USD)", fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
    print(f"Saving plot to market-data.png")     
    plt.savefig(f'market-data.png', dpi=1200)
    
    # pickle data
    print(f"Saving data to market-data.pkl")
    with open('market-data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved")     


if __name__ == "__main__":

    data = get_symbols_list_data(QUERY_TICKER_LIST, START_DATE, END_DATE)
    sns_plots(data, ['Close', 'Volume'], ['line', 'line'], START_DATE, END_DATE)

