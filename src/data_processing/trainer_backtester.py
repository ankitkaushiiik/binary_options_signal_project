# File: src/training/trainer_backtester.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler # Optional but recommended
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Alternative simple model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading model and scaler
import sys
import os
import traceback # Import traceback

# --- Configuration ---
# Adjust path to find the 'src' directory and import the data processor
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from data_processing.processor import fetch_and_process_data
    print("Successfully imported data processor from src/data_processing.")
except ImportError as e:
     print(f"Error: Could not import 'fetch_and_process_data'. Ensure Phase 1 script is in src/data_processing/processor.py")
     print(f"PYTHONPATH: {sys.path}")
     print(f"Import Error: {e}")
     exit()

# --- Model & Data Configuration ---
TICKER = "EURUSD=X"
START_DATE = None
END_DATE = None
INTERVAL = "5m"
PREDICTION_HORIZON_INTERVALS = 1
TRAINING_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "simple_logistic_regression_model.pkl"
SCALER_FILENAME = "standard_scaler.pkl"
MODEL_CHOICE = "logistic_regression"
TEST_SPLITS = 5

# --- Feature Selection ---
# !! CORRECTED !!: Use the EXACT UPPERCASE column names output by pandas-ta
FEATURE_COLS = [
    # MACD features
    'MACD_12_26_9',  # MACD line
    'MACDh_12_26_9', # MACD histogram
    'MACDs_12_26_9',  # MACD signal line
    # RSI features
    'RSI_14',
    # Bollinger Bands features
    'BBL_5_2.0',    # Lower band
    'BBM_5_2.0',    # Middle band (SMA)
    'BBU_5_2.0',    # Upper band
    'BBB_5_2.0',    # Bandwidth
    'BBP_5_2.0'     # Percent B
]

def prepare_features_and_target(df, horizon_intervals=1):
    """Prepares features (X) and target (y) for the ML model."""
    print(f"\nPreparing features and target variable (Horizon: {horizon_intervals} intervals)...")
    if df is None or df.empty:
        print("ERROR: Input DataFrame is empty or None.")
        return None, None

    # Verify all feature columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required feature columns in DataFrame: {missing_cols}")
        print(f"Available columns in DataFrame: {df.columns.tolist()}")
        return None, None
    print("All specified feature columns found.")
    X = df[FEATURE_COLS].copy()

    # Define target variable: 1 if price goes UP after 'horizon' periods, 0 otherwise
    future_close = df['close'].shift(-horizon_intervals)
    price_change = future_close - df['close']
    df['target'] = np.where(price_change > 0, 1, 0)

    original_rows = len(df)
    # Ensure target is also present when dropping NaNs
    df.dropna(subset=FEATURE_COLS + ['target'], inplace=True)
    rows_after_na = len(df)
    print(f"Dropped {original_rows - rows_after_na} rows due to NaN target (shifting) or features.")

    if df.empty:
        print("ERROR: DataFrame became empty after dropping NaN targets.")
        return None, None

    X = df[FEATURE_COLS]
    y = df['target']

    print(f"Features (X) prepared. Shape: {X.shape}")
    print(f"Target (y) prepared. Shape: {y.shape}")
    if not y.empty:
         print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    else:
         print("Warning: Target variable is empty.")

    return X, y

def perform_detailed_backtest(model, scaler, X_test_scaled, y_test):
    """Simulates trading based on model predictions on the test set and prints details."""
    print("\n--- Performing Detailed Backtest Simulation ---")
    if len(X_test_scaled) == 0 or len(y_test) == 0:
         print("Test data is empty, cannot perform backtest.")
         return 0.0

    try:
        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Overall Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report (Test Set):")
        # Added zero_division=0 to handle cases where a class might have no predicted samples
        print(classification_report(y_test, predictions, target_names=['PUT (0)', 'CALL (1)'], zero_division=0))
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_test, predictions))

        win_rate = accuracy * 100
        print(f"\nSimulated Win Rate on Test Set: {win_rate:.2f}%")
        return win_rate
    except Exception as e:
        print(f"ERROR during backtest evaluation: {e}")
        print(traceback.format_exc())
        return 0.0


# --- Main Training and Evaluation Script ---
if __name__ == "__main__":
    print("="*30)
    print("--- Starting Phase 2: Model Training & Backtesting ---")
    print("="*30)

    # 1. Fetch and Process Data
    print("\nStep 1: Fetching and processing data...")
    data = fetch_and_process_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL)

    if data is None:
        print("\nERROR: Failed to get data from processor. Exiting.")
        exit()
    print(f"Data loaded successfully. Shape: {data.shape}")

    # 2. Prepare Features and Target Variable
    X, y = prepare_features_and_target(data, horizon_intervals=PREDICTION_HORIZON_INTERVALS)

    if X is None or y is None or X.empty or y.empty:
         print("\nERROR: Failed to prepare features/target. Exiting.")
         exit()

    # 3. Split Data using TimeSeriesSplit
    print(f"\nStep 3: Splitting data using TimeSeriesSplit (n_splits={TEST_SPLITS})")
    tscv = TimeSeriesSplit(n_splits=TEST_SPLITS)
    X_train, X_test, y_train, y_test = None, None, None, None # Initialize
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f"Split {i+1}: Train indices {len(train_index)}, Test indices {len(test_index)}")

    if X_train is None or X_test is None or X_train.empty or X_test.empty:
        print("\nERROR: Could not split data. Check TimeSeriesSplit settings or data size.")
        exit()

    print(f"\nUsing last split for training and final evaluation:")
    print(f"Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set size:  X={X_test.shape}, y={y_test.shape}")


    # 4. Feature Scaling
    print("\nStep 4: Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    scaler_path = os.path.join(TRAINING_DIR, SCALER_FILENAME)
    try:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved successfully to: {scaler_path}")
    except Exception as e:
        print(f"ERROR: Could not save scaler to {scaler_path}. Error: {e}")


    # 5. Train Model
    print(f"\nStep 5: Training model ({MODEL_CHOICE})...")
    if MODEL_CHOICE == "logistic_regression":
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    elif MODEL_CHOICE == "decision_tree":
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
    else:
         print(f"ERROR: Unknown model choice '{MODEL_CHOICE}'.")
         exit()

    try:
        model.fit(X_train_scaled, y_train)
        print("Model training complete.")
    except Exception as e:
         print(f"ERROR: Model training failed. Error: {e}")
         print(traceback.format_exc())
         exit()

    # 6. Evaluate Model & Perform Backtest Simulation on Test Set
    print("\nStep 6: Evaluating model on the test set...")
    perform_detailed_backtest(model, scaler, X_test_scaled, y_test)

    # 7. Save Trained Model
    print("\nStep 7: Saving the trained model...")
    model_path = os.path.join(TRAINING_DIR, MODEL_FILENAME)
    try:
        joblib.dump(model, model_path)
        print(f"Model saved successfully to: {model_path}")
    except Exception as e:
        print(f"ERROR: Could not save model to {model_path}. Error: {e}")


    print("\n" + "="*30)
    print("--- Phase 2 Finished ---")
    print("="*30)