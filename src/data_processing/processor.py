# File: src/data_processing/processor.py

import pandas as pd
import yfinance as yf
import numpy as np # Import numpy first

# FIX: Define the alias pandas_ta expects before importing it
np.NaN = np.nan

import pandas_ta as ta # Now import pandas_ta
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta # Still used if explicit dates are given
import traceback # For detailed error printing

print("Using pandas-ta for indicators.")

# Load environment variables (optional)
load_dotenv()

def fetch_and_process_data(ticker="EURUSD=X", start_date=None, end_date=None, interval="5m"):
    """
    Fetches historical market data using yfinance, calculates standard technical
    indicators using pandas-ta, and returns a processed DataFrame.

    MODIFIED: If start_date and end_date are None, uses yfinance's 'period'
    parameter to fetch recent data based on interval limits, avoiding reliance
    on the local system clock for the default case. Otherwise, uses the provided
    start_date and end_date.

    Args:
        ticker (str): The ticker symbol compatible with Yahoo Finance (e.g., 'EURUSD=X').
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        interval (str, optional): Data interval (e.g., '1m', '5m', '1h', '1d').
                                  Defaults to "5m".

    Returns:
        pandas.DataFrame or None: A DataFrame containing the fetched OHLCV data
                                  plus calculated indicator columns (MACD, RSI, BBands),
                                  with NaN rows dropped. Returns None on failure.
    """

    use_period_param = False
    yf_period = None
    yf_start = start_date
    yf_end = end_date

    # --- Determine Fetch Method ---
    if start_date is None and end_date is None:
        use_period_param = True
        print(f"INFO: No start/end date provided. Using 'period' parameter based on interval '{interval}'.")
        # Map interval to appropriate yfinance period covering its max history
        if interval == '1m':
            yf_period = "7d" # Max history for 1m is 7 days
        elif interval in ['2m', '5m', '15m', '30m']:
            yf_period = "60d" # Max history is 60 days
        elif interval == '1h':
             # Max history 730 days, use "2y" or specify days if needed, sticking to days for consistency
             # Note: yf might have limits on days for 'period', test this if needed. 60d is safer.
             # Let's use 60d as a safer default even for 1h to avoid potential 'period' issues
             # yf_period = "730d" # This might not work reliably with 'period'
             yf_period = "2y" # Prefer standard periods like 1y, 2y
        else: # Daily or longer
            yf_period = "2y" # Fetch 2 years history for daily+ data by default
        print(f"INFO: Determined yfinance period: '{yf_period}' for interval '{interval}'.")
        yf_start = None # Don't use start/end when using period
        yf_end = None
    elif start_date is not None and end_date is not None:
         print(f"INFO: Using provided start_date '{start_date}' and end_date '{end_date}'.")
         # Validate date formats (optional but good practice)
         try:
             datetime.strptime(start_date, '%Y-%m-%d')
             datetime.strptime(end_date, '%Y-%m-%d')
             yf_start = start_date
             yf_end = end_date
         except ValueError:
             print(f"ERROR: Invalid date format provided. Use 'YYYY-MM-DD'. Cannot fetch data.")
             return None
    else:
         # Handle cases where only one date is provided if necessary, or disallow
         print(f"ERROR: Please provide both start_date and end_date, or neither (to use automatic period).")
         return None


    print(f"Attempting fetch for '{ticker}' | Interval: {interval}"
          f"{' | Period: ' + yf_period if use_period_param else ''}"
          f"{' | Start: ' + yf_start if not use_period_param else ''}"
          f"{' | End: ' + yf_end if not use_period_param else ''}")

    # --- Data Fetching ---
    try:
        df = yf.download(
            tickers=ticker,
            period=yf_period,   # Use period if determined
            start=yf_start,     # Use start if provided (period=None)
            end=yf_end,         # Use end if provided (period=None)
            interval=interval,
            progress=False,
            auto_adjust=False,  # Keep False for raw data
            actions=False
        )

        if df.empty:
            print(f"Error: No data downloaded from yfinance for the specified parameters.")
            # Add more specific error hints if possible
            if use_period_param:
                 print(f"Hint: Check if the ticker '{ticker}' is valid and has data for the period '{yf_period}' at '{interval}' interval.")
            else:
                 print(f"Hint: Check if the ticker '{ticker}' is valid and has data between {yf_start} and {yf_end} at '{interval}' interval.")
            return None

        print(f"Data fetched successfully. Initial Shape: {df.shape}")
        print(f"DEBUG: Original column names: {df.columns.tolist()}")

        # --- Data Cleaning & Validation ---
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns by taking first level only
            df.columns = df.columns.get_level_values(0)

        # Convert column names to strings and lowercase them
        df.columns = df.columns.str.lower()
        print(f"DEBUG: Original column names: {df.columns.tolist()}")
        # Map common yfinance column name variations to our required format
        column_mapping = {
            'adj close': 'adj_close'
        }
        df.rename(columns=column_mapping, inplace=True)
        print(f"DEBUG: Mapped column names: {df.columns.tolist()}")
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             print(f"Error: Missing required OHLCV columns after download/lowercasing: {missing_cols}")
             return None
        print("Required columns verified.")

        # --- Indicator Calculation (using pandas_ta) ---
        print("Calculating indicators using pandas-ta...")
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.bbands(append=True)

        # --- Final Processing ---
        initial_rows = len(df)
        df.dropna(inplace=True)
        rows_after_na = len(df)
        print(f"Indicators calculated. Rows dropped due to NaNs: {initial_rows - rows_after_na}")

        if df.empty:
             print("Error: DataFrame became empty after dropping NaNs from indicator calculation.")
             return None

        print(f"Final DataFrame shape after processing: {df.shape}")
        return df

    except Exception as e:
        print(f"An error occurred during data fetching or processing for {ticker}: {e}")
        print("--- Traceback ---")
        print(traceback.format_exc())
        print("--- End Traceback ---")
        return None

# --- Main execution block for testing the function ---
if __name__ == "__main__":
    print("="*30)
    print("Running Data Processing Test Script")
    print("="*30)

    test_ticker = "EURUSD=X"
    test_interval = "5m" # Test with 5-minute data

    print(f"\nTesting fetch_and_process_data with automatic period (ignores local clock):")
    print(f"  Ticker: {test_ticker}")
    print(f"  Interval: {test_interval}")
    print("-"*30)

    # Call the function WITHOUT start/end dates to trigger 'period' logic
    processed_data = fetch_and_process_data(
        ticker=test_ticker,
        interval=test_interval,
        start_date=None,
        end_date=None
    )

    print("-"*30)
    if processed_data is not None:
        print("Data processing successful!")
        print("\n--- Processed Data Sample (Head) ---")
        print(processed_data.head())
        print("\n--- Processed Data Sample (Tail) ---")
        print(processed_data.tail())
        print("\n--- Data Info ---")
        processed_data.info()
        print("\n--- Available Columns ---")
        print(processed_data.columns.tolist())
    else:
        print("\nData processing failed. Check error messages above.")

    print("\n" + "="*30)
    print("Data Processing Test Finished.")
    print("="*30)

    # --- You can still test with explicit dates if needed ---
    # print("\nTesting with EXPLICIT historical dates (WILL FAIL if dates are in the future):")
    # try:
    #     # Use dates guaranteed to be in the past relative to REAL time
    #     explicit_start = "2024-04-01" # Example PAST date
    #     explicit_end = "2024-04-15"   # Example PAST date
    #     print(f"  Ticker: {test_ticker}")
    #     print(f"  Interval: 1h")
    #     print(f"  Start Date: {explicit_start}")
    #     print(f"  End Date: {explicit_end}")
    #     print("-"*30)
    #     processed_data_explicit = fetch_and_process_data(
    #         ticker=test_ticker,
    #         interval="1h", # Different interval for variety
    #         start_date=explicit_start,
    #         end_date=explicit_end
    #     )
    #     print("-"*30)
    #     if processed_data_explicit is not None:
    #         print("Explicit date processing successful!")
    #         print(processed_data_explicit.head())
    #     else:
    #         print("\nExplicit date processing failed.")
    # except Exception as e:
    #      print(f"Error during explicit date test: {e}")
    # print("\n" + "="*30)