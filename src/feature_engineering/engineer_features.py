# File: src/feature_engineering/engineer_features.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import time
import traceback
from sklearn.linear_model import LinearRegression

# --- Configuration ---
# ...(Keep your configuration the same)...
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'EUR_USD_Historical_Data.csv') 
TIMESTAMP_COL = 'Date'
OPEN_COL = 'Open'
HIGH_COL = 'High'
LOW_COL = 'Low'
CLOSE_COL = 'Price'
VOLUME_COL = None
DATETIME_FORMAT = '%d-%m-%Y'
TARGET_INTERVAL = None
PREDICTION_HORIZON = 1
OUTPUT_DIR = '../../data/'
OUTPUT_FILENAME_BASE = 'eurusd_enriched'
# --- End Configuration ---

# --- Pattern Thresholds ---
# ...(Keep thresholds the same)...
DOJI_THRESHOLD = 0.05
MARUBOZU_WICK_THRESHOLD = 0.05
HAMMER_LIKE_RATIO = 1.8
STAR_BODY_THRESHOLD_RATIO = 0.4
SOLDIER_CROW_BODY_MIN_RATIO = 0.5

# <<< *** FIX: Moved function definition here, BEFORE it's used *** >>>
def calculate_slope(series, window=5):
    """Calculates slope using linear regression over a rolling window."""
    # This function is called by rolling().apply() - it receives a Series segment
    if series.isnull().any() or len(series) < window: # Check for NaNs within the window
        return np.nan
    X = np.arange(window).reshape(-1, 1)
    y = series.values[-window:].reshape(-1, 1) # Use last 'window' points
    try:
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0][0]
    except ValueError:
        return np.nan

def calculate_features(df):
    """Calculates all features on the input DataFrame."""
    print("Calculating features...")
    start_time = time.time()
    df_calc = df.copy() # Work on a copy

    # --- Basic Indicators (pandas_ta) ---
    print("  Calculating basic indicators (pandas_ta)...")
    # ...(Indicator calculation and initial dropna remain the same)...
    try:
        df_calc.ta.macd(append=True)
        df_calc.ta.rsi(append=True)
        df_calc.ta.stoch(append=True)
        df_calc.ta.bbands(length=20, std=2, append=True)
        df_calc.ta.atr(append=True)
        df_calc.ta.sma(length=10, append=True)
        df_calc.ta.sma(length=20, append=True)
        df_calc.ta.sma(length=50, append=True)
        df_calc.ta.ema(length=20, append=True)

        print("\n  --- DEBUG: NaN counts AFTER basic indicator calculation ---")
        print(df_calc.isnull().sum())
        print("  -----------------------------------------------------------\n")
        all_nan_cols = df_calc.columns[df_calc.isnull().all()].tolist()
        if all_nan_cols:
            print(f"  WARN: The following columns are entirely NaN: {all_nan_cols}")

        initial_rows_indicators = len(df_calc)
        df_calc.dropna(inplace=True)
        print(f"    Dropped {initial_rows_indicators - len(df_calc)} rows after initial indicator NaNs.")
        if df_calc.empty:
             print("    ERROR: DataFrame empty after dropping initial indicator NaNs.")
             print("\n  --- DEBUG: Tail of data BEFORE critical dropna() ---")
             print(df.tail(60))
             print("  -----------------------------------------------------\n")
             return None
    except Exception as e:
        print(f"ERROR calculating basic indicators: {e}")
        print(traceback.format_exc())
        return None

    # --- Candlestick Anatomy ---
    # ...(Remains the same)...
    print("  Calculating candlestick anatomy...")
    df_calc['body_size'] = abs(df_calc['open'] - df_calc['close'])
    df_calc['upper_wick'] = df_calc['high'] - np.maximum(df_calc['open'], df_calc['close'])
    df_calc['lower_wick'] = np.minimum(df_calc['open'], df_calc['close']) - df_calc['low']
    df_calc['total_range'] = df_calc['high'] - df_calc['low']
    df_calc['body_to_range_ratio'] = df_calc['body_size'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['upper_wick_ratio'] = df_calc['upper_wick'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['lower_wick_ratio'] = df_calc['lower_wick'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['is_green_candle'] = (df_calc['close'] > df_calc['open']).astype(int)
    df_calc['is_red_candle'] = (df_calc['close'] < df_calc['open']).astype(int)


    # --- Simple Candlestick Patterns ---
    # ...(Remains the same)...
    print("  Calculating simple candlestick patterns...")
    df_calc['is_doji'] = (df_calc['body_to_range_ratio'] < DOJI_THRESHOLD).astype(int)
    df_calc['is_marubozu_bullish'] = ((df_calc['is_green_candle'] == 1) & (df_calc['upper_wick_ratio'] < MARUBOZU_WICK_THRESHOLD) & (df_calc['lower_wick_ratio'] < MARUBOZU_WICK_THRESHOLD)).astype(int)
    df_calc['is_marubozu_bearish'] = ((df_calc['is_red_candle'] == 1) & (df_calc['upper_wick_ratio'] < MARUBOZU_WICK_THRESHOLD) & (df_calc['lower_wick_ratio'] < MARUBOZU_WICK_THRESHOLD)).astype(int)
    df_calc['is_hammer_like'] = ((df_calc['lower_wick'] >= HAMMER_LIKE_RATIO * df_calc['body_size']) & (df_calc['upper_wick_ratio'] < 0.1) & (df_calc['body_size'] > 1e-9)).astype(int)
    df_calc['is_shooting_star_like'] = ((df_calc['upper_wick'] >= HAMMER_LIKE_RATIO * df_calc['body_size']) & (df_calc['lower_wick_ratio'] < 0.1) & (df_calc['body_size'] > 1e-9)).astype(int)

    prev_close = df_calc['close'].shift(1)
    prev_open = df_calc['open'].shift(1)
    prev_low = df_calc['low'].shift(1)
    prev_high = df_calc['high'].shift(1)
    prev_is_green = df_calc['is_green_candle'].shift(1)
    prev_is_red = df_calc['is_red_candle'].shift(1)

    df_calc['is_engulfing_bullish'] = ((df_calc['is_green_candle'] == 1) & (prev_is_red == 1) & (df_calc['open'] < prev_close) & (df_calc['close'] > prev_open)).astype(int)
    df_calc['is_engulfing_bearish'] = ((df_calc['is_red_candle'] == 1) & (prev_is_green == 1) & (df_calc['open'] > prev_close) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_piercing_line'] = ((prev_is_red == 1) & (df_calc['is_green_candle'] == 1) & (df_calc['open'] < prev_low) & (df_calc['close'] > (prev_open + prev_close) / 2) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_dark_cloud'] = ((prev_is_green == 1) & (df_calc['is_red_candle'] == 1) & (df_calc['open'] > prev_high) & (df_calc['close'] < (prev_open + prev_close) / 2) & (df_calc['close'] > prev_open)).astype(int)


    # --- Multi-Candle Patterns ---
    # ...(Remains the same)...
    print("  Calculating multi-candle patterns...")
    df_calc['is_harami_bullish'] = ((prev_is_red == 1) & (df_calc['is_green_candle'] == 1) & (df_calc['open'] > prev_close) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_harami_bearish'] = ((prev_is_green == 1) & (df_calc['is_red_candle'] == 1) & (df_calc['open'] < prev_close) & (df_calc['close'] > prev_open)).astype(int)

    prev2_close = df_calc['close'].shift(2)
    prev2_open = df_calc['open'].shift(2)
    prev2_is_green = df_calc['is_green_candle'].shift(2)
    prev2_is_red = df_calc['is_red_candle'].shift(2)
    prev_body_size = df_calc['body_size'].shift(1)
    prev2_body_size = df_calc['body_size'].shift(2)
    avg_body_size = df_calc['body_size'].rolling(window=20, min_periods=5).mean().shift(1).fillna(method='bfill')

    star_body_cond = prev_body_size < STAR_BODY_THRESHOLD_RATIO * avg_body_size
    morning_star_cond = (prev2_is_red == 1) & star_body_cond & \
                        (np.maximum(prev_open, prev_close) < prev2_close) & \
                        (df_calc['is_green_candle'] == 1) & (df_calc['close'] > (prev2_open + prev2_close)/2)
    df_calc['is_morning_star_like'] = morning_star_cond.astype(int)

    evening_star_cond = (prev2_is_green == 1) & star_body_cond & \
                        (np.minimum(prev_open, prev_close) > prev2_close) & \
                        (df_calc['is_red_candle'] == 1) & (df_calc['close'] < (prev2_open + prev2_close)/2)
    df_calc['is_evening_star_like'] = evening_star_cond.astype(int)

    long_body_cond = df_calc['body_size'] > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO
    prev_long_body_cond = prev_body_size > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO
    prev2_long_body_cond = prev2_body_size > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO

    soldiers_cond = (df_calc['is_green_candle'] == 1) & (prev_is_green == 1) & (prev2_is_green == 1) & \
                    (df_calc['close'] > prev_close) & (prev_close > prev2_close) & \
                    long_body_cond & prev_long_body_cond & prev2_long_body_cond
    df_calc['is_3_white_soldiers_like'] = soldiers_cond.astype(int)

    crows_cond = (df_calc['is_red_candle'] == 1) & (prev_is_red == 1) & (prev2_is_red == 1) & \
                 (df_calc['close'] < prev_close) & (prev_close < prev2_close) & \
                 long_body_cond & prev_long_body_cond & prev2_long_body_cond
    df_calc['is_3_black_crows_like'] = crows_cond.astype(int)


    # --- Indicator Signals & Comparisons ---
    # ...(Remains the same)...
    print("  Calculating indicator signals and comparisons...")
    df_calc['rsi_oversold'] = (df_calc.get('RSI_14', 50) < 30).astype(int)
    df_calc['rsi_overbought'] = (df_calc.get('RSI_14', 50) > 70).astype(int)
    df_calc['stoch_oversold'] = (df_calc.get('STOCHk_14_3_3', 50) < 20).astype(int)
    df_calc['stoch_overbought'] = (df_calc.get('STOCHk_14_3_3', 50) > 80).astype(int)

    if 'STOCHk_14_3_3' in df_calc.columns and 'STOCHd_14_3_3' in df_calc.columns:
        df_calc['stoch_bullish_cross_signal'] = ((df_calc['STOCHk_14_3_3'] > df_calc['STOCHd_14_3_3']) & (df_calc['STOCHk_14_3_3'].shift(1) <= df_calc['STOCHd_14_3_3'].shift(1))).astype(int)
        df_calc['stoch_bearish_cross_signal'] = ((df_calc['STOCHk_14_3_3'] < df_calc['STOCHd_14_3_3']) & (df_calc['STOCHk_14_3_3'].shift(1) >= df_calc['STOCHd_14_3_3'].shift(1))).astype(int)
    else: df_calc['stoch_bullish_cross_signal'] = df_calc['stoch_bearish_cross_signal'] = 0

    if 'MACD_12_26_9' in df_calc.columns and 'MACDs_12_26_9' in df_calc.columns:
        df_calc['macd_bullish_cross_signal'] = ((df_calc['MACD_12_26_9'] > df_calc['MACDs_12_26_9']) & (df_calc['MACD_12_26_9'].shift(1) <= df_calc['MACDs_12_26_9'].shift(1))).astype(int)
        df_calc['macd_bearish_cross_signal'] = ((df_calc['MACD_12_26_9'] < df_calc['MACDs_12_26_9']) & (df_calc['MACD_12_26_9'].shift(1) >= df_calc['MACDs_12_26_9'].shift(1))).astype(int)
    else: df_calc['macd_bullish_cross_signal'] = df_calc['macd_bearish_cross_signal'] = 0

    if 'SMA_10' in df_calc.columns and 'SMA_50' in df_calc.columns:
        df_calc['price_vs_SMA50'] = df_calc['close'] - df_calc['SMA_50']
        df_calc['sma10_cross_sma50_bullish_signal'] = ((df_calc['SMA_10'] > df_calc['SMA_50']) & (df_calc['SMA_10'].shift(1) <= df_calc['SMA_50'].shift(1))).astype(int)
        df_calc['sma10_cross_sma50_bearish_signal'] = ((df_calc['SMA_10'] < df_calc['SMA_50']) & (df_calc['SMA_10'].shift(1) >= df_calc['SMA_50'].shift(1))).astype(int)
    else: df_calc['price_vs_SMA50'] = df_calc['sma10_cross_sma50_bullish_signal'] = df_calc['sma10_cross_sma50_bearish_signal'] = 0

    bbl_col = next((col for col in df_calc.columns if 'BBL_' in col), None)
    bbu_col = next((col for col in df_calc.columns if 'BBU_' in col), None)
    if bbl_col and bbu_col:
        df_calc['price_touch_bbl'] = (df_calc['low'] <= df_calc[bbl_col]).astype(int)
        df_calc['price_touch_bbu'] = (df_calc['high'] >= df_calc[bbu_col]).astype(int)
    else: df_calc['price_touch_bbl'] = df_calc['price_touch_bbu'] = 0


    # --- Volatility ---
    # ...(Remains the same)...
    print("  Calculating volatility features...")
    atr_col = next((col for col in df_calc.columns if 'ATRr_' in col), None)
    if atr_col:
      df_calc['volatility_atr_norm'] = df_calc[atr_col].div(df_calc['close'].replace(0, np.nan)).fillna(0)
    else: df_calc['volatility_atr_norm'] = 0

    bbb_col = next((col for col in df_calc.columns if 'BBB_' in col), None)
    bbm_col = next((col for col in df_calc.columns if 'BBM_' in col), None)
    if bbb_col and bbm_col:
      df_calc['volatility_bbw_norm'] = df_calc[bbb_col].div(df_calc[bbm_col].replace(0, np.nan)).fillna(0)
    else: df_calc['volatility_bbw_norm'] = 0


    # --- S&R / Trend ---
    # ...(Remains the same)...
    print("  Calculating S&R/Trend features...")
    df_calc['rolling_high_20'] = df_calc['high'].shift(1).rolling(window=20, min_periods=5).max()
    df_calc['rolling_low_20'] = df_calc['low'].shift(1).rolling(window=20, min_periods=5).min()
    df_calc['dist_from_high20'] = df_calc['rolling_high_20'] - df_calc['close']
    df_calc['dist_from_low20'] = df_calc['close'] - df_calc['rolling_low_20']

    if 'SMA_50' in df_calc.columns:
        print("    Calculating SMA50 slope...")
        # Apply the slope calculation using a rolling window.
        df_calc['sma50_slope'] = df_calc['SMA_50'].rolling(window=10, min_periods=10).apply(lambda x: calculate_slope(x, window=5), raw=False) # <- Calling the function HERE
    else: df_calc['sma50_slope'] = 0

    if 'SMA_10' in df_calc.columns and 'SMA_50' in df_calc.columns:
        df_calc['trend_dir_sma_confirm'] = 0 # Default 0
        trend_up_cond = (df_calc['SMA_10'] > df_calc['SMA_50']) & (df_calc['SMA_50'] > df_calc['SMA_50'].shift(1))
        trend_down_cond = (df_calc['SMA_10'] < df_calc['SMA_50']) & (df_calc['SMA_50'] < df_calc['SMA_50'].shift(1))
        df_calc.loc[trend_up_cond, 'trend_dir_sma_confirm'] = 1
        df_calc.loc[trend_down_cond, 'trend_dir_sma_confirm'] = -1
    else: df_calc['trend_dir_sma_confirm'] = 0


    # --- Time Features ---
    # ...(Remains the same)...
    print("  Calculating time features...")
    if df_calc.index.tz is None:
        print("  WARN: DataFrame index is timezone-naive. Assuming UTC for time features.")
        try: df_calc.index = df_calc.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        except Exception as tz_err: print(f"  ERROR localizing index: {tz_err}")
    else:
        try: df_calc.index = df_calc.index.tz_convert('UTC')
        except Exception as tz_err: print(f"  ERROR converting index to UTC: {tz_err}")

    df_calc['hour_utc'] = df_calc.index.hour
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['is_london_open'] = ((df_calc['hour_utc'] >= 7) & (df_calc['hour_utc'] < 16)).astype(int)
    df_calc['is_ny_open'] = ((df_calc['hour_utc'] >= 13) & (df_calc['hour_utc'] < 22)).astype(int)
    df_calc['is_london_ny_overlap'] = ((df_calc['hour_utc'] >= 13) & (df_calc['hour_utc'] < 16)).astype(int)


    print(f"  Feature calculation finished. Checkpoint rows: {len(df_calc)}. NaNs per column (sample):\n{df_calc.isnull().sum().tail(30)}")

    print(f"Total feature calculation time: {time.time() - start_time:.2f} seconds.")
    return df_calc


# --- Main Execution ---
# ...(The __main__ block remains the same as the previous corrected version)...
if __name__ == "__main__":
    print("="*30)
    print("--- Starting Phase R2: Feature Engineering ---")
    print("="*30)

    # 1. Load Data
    print(f"Loading historical data from: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at '{DATA_FILE_PATH}'. Please check configuration.")
        exit()
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Data loaded successfully. Initial rows: {len(df)}")

        print(f"Parsing timestamp column '{TIMESTAMP_COL}' with format '{DATETIME_FORMAT}'...")
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format=DATETIME_FORMAT, errors='coerce')
        df.dropna(subset=[TIMESTAMP_COL], inplace=True)
        df.set_index(TIMESTAMP_COL, inplace=True)
        print(f"Timestamp parsing complete. Rows after date parsing: {len(df)}")

    except Exception as e:
        print(f"ERROR: Failed to load or parse CSV file: {e}")
        print(traceback.format_exc())
        exit()

    # 2. Basic Cleaning, Renaming, and Column Selection
    print("Cleaning and renaming columns...")
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.lower()

    rename_map = {
        OPEN_COL.lower(): 'open', HIGH_COL.lower(): 'high',
        LOW_COL.lower(): 'low', CLOSE_COL.lower(): 'close',
    }
    processed_volume = False
    if VOLUME_COL and VOLUME_COL.lower() in df.columns:
        try:
            if df[VOLUME_COL.lower()].dtype == 'object':
                print("  Converting Volume column (handling K/M)...")
                def convert_volume(vol_str):
                    vol_str = str(vol_str).strip().upper()
                    if vol_str == '-' or vol_str == '': return np.nan
                    if 'K' in vol_str: return pd.to_numeric(vol_str.replace('K',''), errors='coerce') * 1e3
                    if 'M' in vol_str: return pd.to_numeric(vol_str.replace('M',''), errors='coerce') * 1e6
                    return pd.to_numeric(vol_str, errors='coerce')
                df['volume_temp'] = df[VOLUME_COL.lower()].apply(convert_volume)
                df.rename(columns={'volume_temp': 'volume'}, inplace=True)
                df.drop(columns=[VOLUME_COL.lower()], inplace=True)
            else:
                 df['volume'] = pd.to_numeric(df[VOLUME_COL.lower()], errors='coerce')
                 if VOLUME_COL.lower() != 'volume':
                     rename_map[VOLUME_COL.lower()] = 'volume'

            if 'volume' in df.columns and df['volume'].isnull().all():
                 print("  WARN: Volume column resulted in all NaNs after conversion. Dropping volume.")
                 df.drop(columns=['volume'], inplace=True)
                 VOLUME_COL = None
            elif 'volume' in df.columns:
                 processed_volume = True

        except Exception as vol_e:
            print(f"  WARN: Error processing volume column '{VOLUME_COL.lower()}': {vol_e}. Dropping volume.")
            if 'volume' in df.columns: df.drop(columns=['volume'], inplace=True)
            if VOLUME_COL.lower() in rename_map: del rename_map[VOLUME_COL.lower()]
            VOLUME_COL = None

    elif VOLUME_COL:
         print(f"WARN: Specified volume column '{VOLUME_COL}' not found in loaded data.")
         VOLUME_COL = None

    df.rename(columns=rename_map, inplace=True)

    required_cols = ['open', 'high', 'low', 'close']
    if processed_volume: required_cols.append('volume')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required standard columns after renaming: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        exit()

    for col in required_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_rows_after_load = len(df)
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    print(f"Dropped {initial_rows_after_load - len(df)} rows with NaN in core OHLC columns.")

    final_cols_to_keep = ['open', 'high', 'low', 'close']
    if processed_volume: final_cols_to_keep.append('volume')

    print(f"Selecting final base columns: {final_cols_to_keep}")
    df_base = df[final_cols_to_keep].copy()

    print(f"Columns cleaned and selected. Using: {df_base.columns.tolist()}")
    print(f"Rows after basic cleaning: {len(df_base)}")

    df_base.sort_index(inplace=True)

    initial_rows = len(df_base)
    df_base = df_base[~df_base.index.duplicated(keep='first')]
    if len(df_base) < initial_rows:
        print(f"Removed {initial_rows - len(df_base)} duplicate timestamp rows.")

    # 3. Resample Data (Optional)
    output_interval_str = "original"
    if TARGET_INTERVAL:
        print(f"Resampling data to '{TARGET_INTERVAL}' interval...")
        try:
            aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            if 'volume' in df_base.columns: aggregation['volume'] = 'sum'
            df_processed = df_base.resample(TARGET_INTERVAL).agg(aggregation)
            df_processed.dropna(subset=['open'], inplace=True)
            print(f"Resampling complete. Rows after resampling: {len(df_processed)}")
            output_interval_str = TARGET_INTERVAL
        except Exception as e:
            print(f"ERROR during resampling: {e}")
            print(traceback.format_exc())
            exit()
    else:
        print("Skipping resampling, using original data interval.")
        df_processed = df_base
        try:
             inferred_freq = pd.infer_freq(df_processed.index)
             output_interval_str = inferred_freq if inferred_freq != 'B' else 'daily'
             if output_interval_str is None: output_interval_str = 'unknown'
             print(f"  Inferred data frequency: {inferred_freq} -> Using '{output_interval_str}' for filename")
        except Exception: output_interval_str = "daily"


    if df_processed.empty:
        print("ERROR: DataFrame is empty after loading/cleaning/resampling. Cannot proceed.")
        exit()

    # 4. Calculate Features
    df_enriched = calculate_features(df_processed.copy())

    if df_enriched is None or df_enriched.empty:
        print("ERROR: Feature calculation failed or resulted in empty DataFrame. Exiting.")
        exit()

    # 5. Define Target Variable
    print("Defining target variable...")
    df_enriched['target'] = np.where(df_enriched['close'].shift(-PREDICTION_HORIZON) > df_enriched['close'], 1, 0)

    # 6. Final NaN Drop for target shift
    print(f"Dropping final {PREDICTION_HORIZON} row(s) due to target shifting...")
    initial_rows = len(df_enriched)
    if 'target' in df_enriched.columns:
        df_enriched.dropna(subset=['target'], inplace=True)
        print(f"Dropped {initial_rows - len(df_enriched)} rows based on target NaNs.")
    else:
        print("WARN: 'target' column not found before final dropna.")

    print(f"Final dataset shape: {df_enriched.shape}")

    if df_enriched.empty:
        print("ERROR: Final DataFrame is empty after target NaN removal.")
        exit()

    # 7. Save Enriched Data
    final_output_filename = f'{OUTPUT_FILENAME_BASE}_{output_interval_str}.parquet'
    output_path = os.path.join(OUTPUT_DIR, final_output_filename)
    print(f"Saving enriched data to: {output_path}")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_enriched.columns = df_enriched.columns.astype(str)
        df_enriched.to_parquet(output_path, index=True)
        print("Data saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save data to Parquet file: {e}")
        print(traceback.format_exc())

    print("\n" + "="*30)
    print("--- Phase R2: Feature Engineering Finished ---")
    print(f"--- Enriched dataset saved to: {output_path} ---")
    print("--- Ready for Phase R3 (Model Selection & Backtesting) ---")
    print("="*30)