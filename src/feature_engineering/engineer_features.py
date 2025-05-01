# File: src/feature_engineering/engineer_features.py
# <<< VERSION FOR COMBINING DUKASCOPY 5min CSVs + LAGGED FEATURES >>>

import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import time
import traceback
from sklearn.linear_model import LinearRegression
import glob # To find files

# --- Configuration --- <<< *** EDIT THIS SECTION *** >>> ---

# 1. Directory containing your downloaded Dukascopy CSV files
#    (Relative path from this script's location)
INPUT_CSV_DIR = '../../data/dukascopy_csvs/' # <<< *** CHANGE THIS PATH ***

# 2. File pattern to match (if needed, otherwise loads all CSVs in dir)
#    Example: 'EURUSD_Candlestick_5_M_BID*.csv'
INPUT_CSV_PATTERN = 'EURUSD_Candlestick_5_M_BID*.csv' # <<< Adjust if filenames differ

# 3. Column names in your CSV files (CASE SENSITIVE!)
TIMESTAMP_COL = 'Gmt time'
OPEN_COL = 'Open'
HIGH_COL = 'High'
LOW_COL = 'Low'
CLOSE_COL = 'Close'
VOLUME_COL = 'Volume' # Volume column exists based on sample

# 4. Datetime format of TIMESTAMP_COL
#    Format like: 01.05.2016 00:00:00.000
DATETIME_FORMAT = '%d.%m.%Y %H:%M:%S.%f' # <<< Verify this matches exactly

# 5. Target interval for features - Data is already 5min, no resampling needed
TARGET_INTERVAL = None

# 6. Prediction horizon (in number of 5-minute intervals)
PREDICTION_HORIZON = 1 # Predict next 5-minute bar

# 7. Output directory and filename for the enriched data
OUTPUT_DIR = '../../data/' # Save processed data in project_root/data/
OUTPUT_FILENAME_BASE = 'eurusd_dukascopy_enriched'
OUTPUT_INTERVAL_STR = '5m' # Explicitly state 5m for output filename

# --- End Configuration ---

# --- Pattern Thresholds ---
DOJI_THRESHOLD = 0.05; MARUBOZU_WICK_THRESHOLD = 0.05; HAMMER_LIKE_RATIO = 1.8
STAR_BODY_THRESHOLD_RATIO = 0.4; SOLDIER_CROW_BODY_MIN_RATIO = 0.5

# --- Helper Function ---
def calculate_slope(series, window=5):
    """Calculates slope using linear regression over a rolling window."""
    if series.isnull().any() or len(series) < window: return np.nan
    X = np.arange(window).reshape(-1, 1); y = series.values[-window:].reshape(-1, 1)
    try: model = LinearRegression(); model.fit(X, y); return model.coef_[0][0]
    except ValueError: return np.nan

# --- Feature Calculation Function ---
# (This function remains IDENTICAL to the previous version that added lagged features)
def calculate_features(df):
    """Calculates all features on the input DataFrame."""
    print("Calculating features...")
    start_time = time.time()
    df_calc = df.copy()
    # --- Basic Indicators (pandas_ta) ---
    print("  Calculating basic indicators (pandas_ta)...")
    try:
        df_calc.ta.macd(append=True); df_calc.ta.rsi(append=True); df_calc.ta.stoch(append=True)
        df_calc.ta.bbands(length=20, std=2, append=True); df_calc.ta.atr(append=True)
        df_calc.ta.sma(length=10, append=True); df_calc.ta.sma(length=20, append=True)
        df_calc.ta.sma(length=50, append=True); df_calc.ta.ema(length=20, append=True)
        initial_rows_indicators = len(df_calc); df_calc.dropna(inplace=True)
        print(f"    Dropped {initial_rows_indicators - len(df_calc)} rows after initial indicator NaNs.")
        if df_calc.empty: print("    ERROR: DataFrame empty after dropping initial indicator NaNs."); return None
    except Exception as e: print(f"ERROR calculating basic indicators: {e}"); print(traceback.format_exc()); return None
    # --- Candlestick Anatomy ---
    print("  Calculating candlestick anatomy...")
    df_calc['body_size'] = abs(df_calc['open'] - df_calc['close']); df_calc['upper_wick'] = df_calc['high'] - np.maximum(df_calc['open'], df_calc['close'])
    df_calc['lower_wick'] = np.minimum(df_calc['open'], df_calc['close']) - df_calc['low']; df_calc['total_range'] = df_calc['high'] - df_calc['low']
    df_calc['body_to_range_ratio'] = df_calc['body_size'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['upper_wick_ratio'] = df_calc['upper_wick'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['lower_wick_ratio'] = df_calc['lower_wick'].div(df_calc['total_range'].replace(0, np.nan)).fillna(0)
    df_calc['is_green_candle'] = (df_calc['close'] > df_calc['open']).astype(int); df_calc['is_red_candle'] = (df_calc['close'] < df_calc['open']).astype(int)
    # --- Simple Candlestick Patterns ---
    print("  Calculating simple candlestick patterns...")
    df_calc['is_doji'] = (df_calc['body_to_range_ratio'] < DOJI_THRESHOLD).astype(int)
    df_calc['is_marubozu_bullish'] = ((df_calc['is_green_candle'] == 1) & (df_calc['upper_wick_ratio'] < MARUBOZU_WICK_THRESHOLD) & (df_calc['lower_wick_ratio'] < MARUBOZU_WICK_THRESHOLD)).astype(int)
    df_calc['is_marubozu_bearish'] = ((df_calc['is_red_candle'] == 1) & (df_calc['upper_wick_ratio'] < MARUBOZU_WICK_THRESHOLD) & (df_calc['lower_wick_ratio'] < MARUBOZU_WICK_THRESHOLD)).astype(int)
    df_calc['is_hammer_like'] = ((df_calc['lower_wick'] >= HAMMER_LIKE_RATIO * df_calc['body_size']) & (df_calc['upper_wick_ratio'] < 0.1) & (df_calc['body_size'] > 1e-9)).astype(int)
    df_calc['is_shooting_star_like'] = ((df_calc['upper_wick'] >= HAMMER_LIKE_RATIO * df_calc['body_size']) & (df_calc['lower_wick_ratio'] < 0.1) & (df_calc['body_size'] > 1e-9)).astype(int)
    prev_close = df_calc['close'].shift(1); prev_open = df_calc['open'].shift(1); prev_low = df_calc['low'].shift(1); prev_high = df_calc['high'].shift(1)
    prev_is_green = df_calc['is_green_candle'].shift(1); prev_is_red = df_calc['is_red_candle'].shift(1)
    df_calc['is_engulfing_bullish'] = ((df_calc['is_green_candle'] == 1) & (prev_is_red == 1) & (df_calc['open'] < prev_close) & (df_calc['close'] > prev_open)).astype(int)
    df_calc['is_engulfing_bearish'] = ((df_calc['is_red_candle'] == 1) & (prev_is_green == 1) & (df_calc['open'] > prev_close) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_piercing_line'] = ((prev_is_red == 1) & (df_calc['is_green_candle'] == 1) & (df_calc['open'] < prev_low) & (df_calc['close'] > (prev_open + prev_close) / 2) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_dark_cloud'] = ((prev_is_green == 1) & (df_calc['is_red_candle'] == 1) & (df_calc['open'] > prev_high) & (df_calc['close'] < (prev_open + prev_close) / 2) & (df_calc['close'] > prev_open)).astype(int)
    # --- Multi-Candle Patterns ---
    print("  Calculating multi-candle patterns...")
    df_calc['is_harami_bullish'] = ((prev_is_red == 1) & (df_calc['is_green_candle'] == 1) & (df_calc['open'] > prev_close) & (df_calc['close'] < prev_open)).astype(int)
    df_calc['is_harami_bearish'] = ((prev_is_green == 1) & (df_calc['is_red_candle'] == 1) & (df_calc['open'] < prev_close) & (df_calc['close'] > prev_open)).astype(int)
    prev2_close = df_calc['close'].shift(2); prev2_open = df_calc['open'].shift(2); prev2_is_green = df_calc['is_green_candle'].shift(2); prev2_is_red = df_calc['is_red_candle'].shift(2)
    prev_body_size = df_calc['body_size'].shift(1); prev2_body_size = df_calc['body_size'].shift(2)
    avg_body_size = df_calc['body_size'].rolling(window=20, min_periods=5).mean().shift(1).bfill()
    star_body_cond = prev_body_size < STAR_BODY_THRESHOLD_RATIO * avg_body_size
    morning_star_cond = (prev2_is_red == 1) & star_body_cond & (np.maximum(prev_open, prev_close) < prev2_close) & (df_calc['is_green_candle'] == 1) & (df_calc['close'] > (prev2_open + prev2_close)/2)
    df_calc['is_morning_star_like'] = morning_star_cond.astype(int)
    evening_star_cond = (prev2_is_green == 1) & star_body_cond & (np.minimum(prev_open, prev_close) > prev2_close) & (df_calc['is_red_candle'] == 1) & (df_calc['close'] < (prev2_open + prev2_close)/2)
    df_calc['is_evening_star_like'] = evening_star_cond.astype(int)
    long_body_cond = df_calc['body_size'] > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO
    prev_long_body_cond = prev_body_size > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO
    prev2_long_body_cond = prev2_body_size > avg_body_size * SOLDIER_CROW_BODY_MIN_RATIO
    soldiers_cond = (df_calc['is_green_candle'] == 1) & (prev_is_green == 1) & (prev2_is_green == 1) & (df_calc['close'] > prev_close) & (prev_close > prev2_close) & long_body_cond & prev_long_body_cond & prev2_long_body_cond
    df_calc['is_3_white_soldiers_like'] = soldiers_cond.astype(int)
    crows_cond = (df_calc['is_red_candle'] == 1) & (prev_is_red == 1) & (prev2_is_red == 1) & (df_calc['close'] < prev_close) & (prev_close < prev2_close) & long_body_cond & prev_long_body_cond & prev2_long_body_cond
    df_calc['is_3_black_crows_like'] = crows_cond.astype(int)
    # --- Indicator Signals & Comparisons ---
    print("  Calculating indicator signals and comparisons...")
    df_calc['rsi_oversold'] = (df_calc.get('RSI_14', 50) < 30).astype(int); df_calc['rsi_overbought'] = (df_calc.get('RSI_14', 50) > 70).astype(int)
    df_calc['stoch_oversold'] = (df_calc.get('STOCHk_14_3_3', 50) < 20).astype(int); df_calc['stoch_overbought'] = (df_calc.get('STOCHk_14_3_3', 50) > 80).astype(int)
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
    bbl_col = next((col for col in df_calc.columns if 'BBL_' in col), None); bbu_col = next((col for col in df_calc.columns if 'BBU_' in col), None)
    if bbl_col and bbu_col: df_calc['price_touch_bbl'] = (df_calc['low'] <= df_calc[bbl_col]).astype(int); df_calc['price_touch_bbu'] = (df_calc['high'] >= df_calc[bbu_col]).astype(int)
    else: df_calc['price_touch_bbl'] = df_calc['price_touch_bbu'] = 0
    # --- Volatility ---
    print("  Calculating volatility features...")
    atr_col = next((col for col in df_calc.columns if 'ATRr_' in col), None)
    if atr_col: df_calc['volatility_atr_norm'] = df_calc[atr_col].div(df_calc['close'].replace(0, np.nan)).fillna(0)
    else: df_calc['volatility_atr_norm'] = 0
    bbb_col = next((col for col in df_calc.columns if 'BBB_' in col), None); bbm_col = next((col for col in df_calc.columns if 'BBM_' in col), None)
    if bbb_col and bbm_col: df_calc['volatility_bbw_norm'] = df_calc[bbb_col].div(df_calc[bbm_col].replace(0, np.nan)).fillna(0)
    else: df_calc['volatility_bbw_norm'] = 0
    # --- S&R / Trend ---
    print("  Calculating S&R/Trend features...")
    df_calc['rolling_high_20'] = df_calc['high'].shift(1).rolling(window=20, min_periods=5).max()
    df_calc['rolling_low_20'] = df_calc['low'].shift(1).rolling(window=20, min_periods=5).min()
    df_calc['dist_from_high20'] = df_calc['rolling_high_20'] - df_calc['close']
    df_calc['dist_from_low20'] = df_calc['close'] - df_calc['rolling_low_20']
    if 'SMA_50' in df_calc.columns: print("    Calculating SMA50 slope..."); df_calc['sma50_slope'] = df_calc['SMA_50'].rolling(window=10, min_periods=10).apply(calculate_slope, kwargs=dict(window=5), raw=False)
    else: df_calc['sma50_slope'] = 0
    if 'SMA_10' in df_calc.columns and 'SMA_50' in df_calc.columns:
        df_calc['trend_dir_sma_confirm'] = 0; trend_up_cond = (df_calc['SMA_10'] > df_calc['SMA_50']) & (df_calc['SMA_50'] > df_calc['SMA_50'].shift(1))
        trend_down_cond = (df_calc['SMA_10'] < df_calc['SMA_50']) & (df_calc['SMA_50'] < df_calc['SMA_50'].shift(1))
        df_calc.loc[trend_up_cond, 'trend_dir_sma_confirm'] = 1; df_calc.loc[trend_down_cond, 'trend_dir_sma_confirm'] = -1
    else: df_calc['trend_dir_sma_confirm'] = 0
    # --- Add Lagged Features ---
    print("  Calculating lagged features...")
    lag_periods = [1, 2, 3, 5]; features_to_lag = ['RSI_14', 'MACDh_12_26_9', 'ATRr_14', 'volatility_atr_norm', 'sma50_slope', 'BBP_20_2.0', 'STOCHk_14_3_3']
    for feature in features_to_lag:
        if feature in df_calc.columns:
            for lag in lag_periods: df_calc[f"{feature}_lag{lag}"] = df_calc[feature].shift(lag)
        else: print(f"    WARN: Cannot lag '{feature}', column not found.")
    initial_rows_lags = len(df_calc); df_calc.dropna(inplace=True) # Drop NaNs from lags
    print(f"    Dropped {initial_rows_lags - len(df_calc)} rows after adding lagged features.")
    if df_calc.empty: print("    ERROR: DataFrame empty after dropping lagged feature NaNs."); return None
    # --- Time Features ---
    print("  Calculating time features...")
    if df_calc.index.tz is None: print("  WARN: DataFrame index is timezone-naive. Assuming UTC."); df_calc.index = df_calc.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
    else: df_calc.index = df_calc.index.tz_convert('UTC')
    df_calc['hour_utc'] = df_calc.index.hour; df_calc['day_of_week'] = df_calc.index.dayofweek # Monday=0
    df_calc['is_london_open'] = ((df_calc['hour_utc'] >= 7) & (df_calc['hour_utc'] < 16)).astype(int)
    df_calc['is_ny_open'] = ((df_calc['hour_utc'] >= 13) & (df_calc['hour_utc'] < 22)).astype(int)
    df_calc['is_london_ny_overlap'] = ((df_calc['hour_utc'] >= 13) & (df_calc['hour_utc'] < 16)).astype(int)
    print(f"  Feature calculation finished. Final Checkpoint rows: {len(df_calc)}")
    print(f"Total feature calculation time: {time.time() - start_time:.2f} seconds.")
    return df_calc

# --- Main Execution ---
if __name__ == "__main__":
    print("="*30); print("--- Starting Phase R2: Feature Engineering (Dukascopy 5min CSVs) ---"); print("="*30)

    # 1. Find and Load CSV Files
    print(f"Looking for CSV files in: {INPUT_CSV_DIR}")
    csv_files = glob.glob(os.path.join(INPUT_CSV_DIR, INPUT_CSV_PATTERN))
    csv_files.sort() # Sort files chronologically based on name assumption

    if not csv_files:
        print(f"ERROR: No CSV files found matching pattern '{INPUT_CSV_PATTERN}' in directory '{INPUT_CSV_DIR}'.")
        exit()

    print(f"Found {len(csv_files)} CSV files to process:")
    for f in csv_files: print(f"  - {os.path.basename(f)}")

    all_data_frames = []
    for i, file_path in enumerate(csv_files):
        print(f"\nLoading and parsing file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}...")
        try:
            df_chunk = pd.read_csv(file_path)
            # Combine date and time columns - Adjust column names if needed
            datetime_col_str = df_chunk[TIMESTAMP_COL].astype(str) # Assuming GMT Time column exists
            # Parse using the specific format
            df_chunk['datetime_parsed'] = pd.to_datetime(datetime_col_str, format=DATETIME_FORMAT, errors='coerce')
            df_chunk.dropna(subset=['datetime_parsed'], inplace=True)
            df_chunk.set_index('datetime_parsed', inplace=True)
            print(f"  Parsed {len(df_chunk)} rows.")
            all_data_frames.append(df_chunk)
        except KeyError as ke:
             print(f"  ERROR: Column not found in {os.path.basename(file_path)}. Expected '{TIMESTAMP_COL}'. Error: {ke}")
             continue # Skip this file
        except Exception as e:
            print(f"  ERROR: Failed to load or parse CSV file {os.path.basename(file_path)}: {e}")
            print(traceback.format_exc())
            continue # Skip this file

    if not all_data_frames:
        print("ERROR: No data could be loaded from any CSV file. Exiting.")
        exit()

    # Concatenate all loaded data
    print("\nConcatenating loaded DataFrames...")
    df = pd.concat(all_data_frames)
    print(f"Concatenated data shape: {df.shape}")

    # 2. Basic Cleaning, Renaming, and Column Selection
    print("Cleaning and renaming columns...")
    df.columns = df.columns.str.lower()
    rename_map = {OPEN_COL.lower(): 'open', HIGH_COL.lower(): 'high', LOW_COL.lower(): 'low', CLOSE_COL.lower(): 'close'}
    processed_volume = False
    if VOLUME_COL and VOLUME_COL.lower() in df.columns:
        rename_map[VOLUME_COL.lower()] = 'volume'
        if df[VOLUME_COL.lower()].dtype == 'object':
            print(f"  WARN: Volume column '{VOLUME_COL}' is object type, attempting conversion...")
            df['volume'] = pd.to_numeric(df[VOLUME_COL.lower()], errors='coerce')
        else: df['volume'] = df[VOLUME_COL.lower()] # Assume numeric
        if df['volume'].isnull().all(): print("  WARN: Volume column all NaNs. Ignoring."); processed_volume = False; df.drop(columns=['volume'], inplace=True)
        else: processed_volume = True
    elif VOLUME_COL: print(f"WARN: Specified volume column '{VOLUME_COL}' not found.")
    df.rename(columns=rename_map, inplace=True)
    required_cols = ['open', 'high', 'low', 'close']
    if processed_volume: required_cols.append('volume')
    missing_cols = [col for col in required_cols if col not in df.columns];
    if missing_cols: print(f"ERROR: Missing required standard columns: {missing_cols}"); exit()
    for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    initial_rows_after_load = len(df); df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    print(f"Dropped {initial_rows_after_load - len(df)} rows with NaN in core OHLC columns.")
    final_cols_to_keep = ['open', 'high', 'low', 'close'];
    if processed_volume: final_cols_to_keep.append('volume')
    print(f"Selecting final base columns: {final_cols_to_keep}")
    df_base = df[final_cols_to_keep].copy()
    print(f"Columns cleaned and selected. Using: {df_base.columns.tolist()}")
    print(f"Rows after basic cleaning: {len(df_base)}")
    df_base.sort_index(inplace=True)
    initial_rows = len(df_base); df_base = df_base[~df_base.index.duplicated(keep='first')]
    if len(df_base) < initial_rows: print(f"Removed {initial_rows - len(df_base)} duplicate timestamp rows.")

    # 3. Resample Data (Skipped)
    print("Skipping resampling, using original 5-minute interval.")
    df_processed = df_base
    output_interval_str = OUTPUT_INTERVAL_STR # Use defined 5m string
    if df_processed.empty: print("ERROR: DataFrame empty after loading/cleaning. Cannot proceed."); exit()

    # 4. Calculate Features (including lags)
    df_enriched = calculate_features(df_processed.copy())
    if df_enriched is None or df_enriched.empty: print("ERROR: Feature calculation failed. Exiting."); exit()

    # 5. Define Target Variable
    print("Defining target variable...")
    df_enriched['target'] = np.where(df_enriched['close'].shift(-PREDICTION_HORIZON) > df_enriched['close'], 1, 0)

    # 6. Final NaN Drop for target shift
    print(f"Dropping final {PREDICTION_HORIZON} row(s) due to target shifting...")
    initial_rows = len(df_enriched)
    if 'target' in df_enriched.columns: df_enriched.dropna(subset=['target'], inplace=True); print(f"Dropped {initial_rows - len(df_enriched)} rows based on target NaNs.")
    else: print("WARN: 'target' column not found before final dropna.")
    print(f"Final dataset shape: {df_enriched.shape}")
    if df_enriched.empty: print("ERROR: Final DataFrame is empty after target NaN removal."); exit()

    # 7. Save Enriched Data
    final_output_filename = f'{OUTPUT_FILENAME_BASE}_{output_interval_str}_with_lags.parquet' # Add suffix
    output_path = os.path.join(OUTPUT_DIR, final_output_filename)
    print(f"Saving enriched 5min Dukascopy data with lags to: {output_path}")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_enriched.columns = df_enriched.columns.astype(str)
        df_enriched.to_parquet(output_path, index=True)
        print("Data saved successfully.")
    except Exception as e: print(f"ERROR: Failed to save data to Parquet file: {e}"); print(traceback.format_exc())

    print("\n" + "="*30)
    print("--- Phase R2: Feature Engineering (Dukascopy 5min) Finished ---")
    print(f"--- Enriched dataset saved to: {output_path} ---")
    print("--- Ready for Phase R3 (Backtesting on Dukascopy 5min data) ---")
    print("="*30)