# # File: src/training/run_backtesting.py

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
# import joblib
# import os
# import time
# import traceback
# import warnings

# # Optional: Import advanced models if installed
# try:
#     import xgboost as xgb
#     XGB_AVAILABLE = True
#     print("XGBoost library found.")
# except ImportError:
#     XGB_AVAILABLE = False
#     print("XGBoost library not found, XGBoost model will be skipped.")

# try:
#     import lightgbm as lgb
#     LGBM_AVAILABLE = True
#     print("LightGBM library found.")
# except ImportError:
#     LGBM_AVAILABLE = False
#     print("LightGBM library not found, LightGBM model will be skipped.")


# # --- Configuration --- <<< REVIEW THIS SECTION >>> ---

# # 1. Path to the enriched data file created in Phase R2
# #    Adjust path relative to where you run this script
# DATA_DIR = '../../data/'
# # Construct filename based on Phase R2 output logic (assuming daily)
# DATA_FILENAME = 'eurusd_enriched_daily.parquet' # Adjust interval string if needed
# ENRICHED_DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

# # 2. Target column name
# TARGET_COL = 'target'

# # 3. Features to EXCLUDE (if any). Leave empty to use all columns except target.
# #    Examples: ['open', 'high', 'low', 'close', 'adj_close'] if you want to exclude basic prices
# #    Or specific indicators you want to test without: ['SMA_10', 'SMA_50']
# EXCLUDE_COLS = ['open', 'high', 'low', 'close'] # Let's exclude raw prices for now

# # 4. TimeSeriesSplit Configuration
# N_SPLITS = 5  # Number of folds for cross-validation/backtesting
# # Optional: Max train size per split to simulate limited history
# # MAX_TRAIN_SIZE = None
# # Optional: Gap between train and test set (e.g., to avoid lookahead bias if target leaks)
# # GAP_DAYS = 0

# # 5. Models to evaluate
# #    (Script will automatically skip XGB/LGBM if libraries not found)
# MODELS_TO_RUN = [
#     'logistic_regression',
#     'random_forest',
#     'gradient_boosting',
#     'xgboost',
#     'lightgbm'
# ]

# # 6. Directory to save best models/results (optional)
# RESULTS_DIR = './backtesting_results' # Subdirectory within src/training

# # --- End Configuration ---


# def load_data(filepath):
#     # ... (load_data function remains the same) ...
#     print(f"\nLoading enriched data from: {filepath}")
#     if not os.path.exists(filepath):
#         print(f"ERROR: Enriched data file not found at '{filepath}'.")
#         print("Please run the Phase R2 feature engineering script first.")
#         return None
#     try:
#         df = pd.read_parquet(filepath)
#         df.index = pd.to_datetime(df.index)
#         df.sort_index(inplace=True)
#         print(f"Data loaded successfully. Shape: {df.shape}")
#         return df
#     except Exception as e:
#         print(f"ERROR: Failed to load Parquet file: {e}")
#         print(traceback.format_exc())
#         return None

# def get_model(model_name):
#     # ... (get_model function remains the same) ...
#     if model_name == 'logistic_regression':
#         return LogisticRegression(random_state=42, class_weight='balanced', max_iter=1500, solver='liblinear')
#     elif model_name == 'random_forest':
#         return RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=10, n_jobs=-1)
#     elif model_name == 'gradient_boosting':
#         return GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
#     elif model_name == 'xgboost' and XGB_AVAILABLE:
#         return xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
#     elif model_name == 'lightgbm' and LGBM_AVAILABLE:
#         return lgb.LGBMClassifier(random_state=42, objective='binary', metric='binary_logloss')
#     else:
#         return None

# def run_backtest_fold(model_name, X_train, y_train, X_test, y_test):
#     # ... (run_backtest_fold function remains the same) ...
#     fold_results = {}
#     model = get_model(model_name)
#     if model is None:
#         print(f"Skipping model '{model_name}' (not available or unknown).")
#         return None

        

#     print(f"  Training {model_name}...")
#     start_train_time = time.time()
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     with warnings.catch_warnings():
#          warnings.simplefilter("ignore")
#          model.fit(X_train_scaled, y_train)

#     train_time = time.time() - start_train_time
#     print(f"  Training finished in {train_time:.2f}s.")

#     print(f"  Evaluating {model_name}...")
#     start_eval_time = time.time()
#     y_pred = model.predict(X_test_scaled)
#     eval_time = time.time() - start_eval_time

#     fold_results['accuracy'] = accuracy_score(y_test, y_pred)
#     fold_results['precision'] = precision_score(y_test, y_pred, zero_division=0)
#     fold_results['recall'] = recall_score(y_test, y_pred, zero_division=0)
#     fold_results['f1'] = f1_score(y_test, y_pred, zero_division=0)
#     fold_results['report'] = classification_report(y_test, y_pred, target_names=['PUT (0)', 'CALL (1)'], zero_division=0, output_dict=True)
#     fold_results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
#     fold_results['train_time'] = train_time
#     fold_results['eval_time'] = eval_time

#     print(f"  Evaluation finished in {eval_time:.2f}s. Accuracy: {fold_results['accuracy']:.4f}")
#     return fold_results


# # --- Main Backtesting Execution ---
# if __name__ == "__main__":
#     print("="*30)
#     print("--- Starting Phase R3: Model Selection & Backtesting ---")
#     print("="*30)

#     # 1. Load Data
#     df = load_data(ENRICHED_DATA_PATH)
#     if df is None:
#         exit()

#     # 2. Prepare Features (X) and Target (y)
#     print("\nPreparing features and target...")
#     if TARGET_COL not in df.columns:
#         print(f"ERROR: Target column '{TARGET_COL}' not found in the DataFrame.")
#         exit()
#     y = df[TARGET_COL]
#     features_to_use = [col for col in df.columns if col != TARGET_COL and col not in EXCLUDE_COLS]
#     print(f"Using {len(features_to_use)} features:")
#     print(f"  {features_to_use[:5]} ... {features_to_use[-5:]}")
#     X = df[features_to_use]

#     # --- *** FIX: Drop remaining NaNs AFTER selecting X and y *** ---
#     print("\nDropping rows with NaNs from features (e.g., rolling windows, slope)...")
#     initial_rows = len(X)
#     # Drop rows in X with any NaNs
#     X = X.dropna()
#     # Align y to match the remaining rows in X using the index
#     y = y.loc[X.index]
#     print(f"Dropped {initial_rows - len(X)} rows.")
#     print(f"Final shape for backtesting: X={X.shape}, y={y.shape}")
#     # --- End NaN Drop Fix ---

#     if X.empty or y.empty:
#         print("ERROR: DataFrame became empty after final NaN drop. Check feature calculations.")
#         exit()

#     # Detailed check for NaNs/Infs after final drop (should pass now)
#     print("\nPerforming final check for NaNs/Infs in features (X)...")
#     nan_check = X.isnull().sum()
#     inf_check = np.isinf(X.values).sum(axis=0)
#     inf_check_series = pd.Series(inf_check, index=X.columns)
#     columns_with_nan = nan_check[nan_check > 0]
#     columns_with_inf = inf_check_series[inf_check_series > 0]

#     if not columns_with_nan.empty or not columns_with_inf.empty:
#          print("ERROR: NaNs or Infs still detected after final dropna. Investigation needed.")
#          if not columns_with_nan.empty: print("Columns with NaNs:\n", columns_with_nan)
#          if not columns_with_inf.empty: print("Columns with Infs:\n", columns_with_inf)
#          exit()
#     else:
#          print("Final check passed: No NaNs or Infs found in feature columns.")


#     # 3. Setup TimeSeriesSplit
#     tscv = TimeSeriesSplit(n_splits=N_SPLITS)
#     print(f"\nUsing TimeSeriesSplit with {N_SPLITS} splits.")

#     # 4. Run Backtesting Loop for each model
#     # ... (Rest of the script remains the same: loop through models, folds, aggregate results) ...
#     all_results = {}

#     for model_name in MODELS_TO_RUN:
#         if model_name == 'xgboost' and not XGB_AVAILABLE: continue
#         if model_name == 'lightgbm' and not LGBM_AVAILABLE: continue

#         print(f"\n--- Evaluating Model: {model_name} ---")
#         model_fold_results = []
#         fold_num = 0

#         for train_index, test_index in tscv.split(X): # Use cleaned X here
#             fold_num += 1
#             print(f"\nProcessing Fold {fold_num}/{N_SPLITS}...")
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Use aligned y here
#             print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
#             print(f"  Train period: {X_train.index.min()} -> {X_train.index.max()}")
#             print(f"  Test period:  {X_test.index.min()} -> {X_test.index.max()}")

#             if X_train.empty or X_test.empty:
#                  print("WARN: Empty train or test set in this fold, skipping.")
#                  continue

#             fold_result = run_backtest_fold(model_name, X_train, y_train, X_test, y_test)
#             if fold_result:
#                  model_fold_results.append(fold_result)

#         all_results[model_name] = model_fold_results

#     # 5. Aggregate and Print Results
#     # ... (remains the same) ...
#     print("\n" + "="*40)
#     print("--- Backtesting Summary ---")
#     print("="*40)

#     summary_stats = {}
#     for model_name, fold_results_list in all_results.items():
#         if not fold_results_list:
#             print(f"\nModel: {model_name} - No results (skipped or failed all folds).")
#             continue

#         print(f"\nModel: {model_name} ({len(fold_results_list)} folds)")
#         avg_accuracy = np.mean([res['accuracy'] for res in fold_results_list])
#         avg_precision = np.mean([res['precision'] for res in fold_results_list])
#         avg_recall = np.mean([res['recall'] for res in fold_results_list])
#         avg_f1 = np.mean([res['f1'] for res in fold_results_list])
#         avg_train_time = np.mean([res['train_time'] for res in fold_results_list])
#         avg_eval_time = np.mean([res['eval_time'] for res in fold_results_list])

#         print(f"  Average Accuracy:  {avg_accuracy:.4f}")
#         print(f"  Average Precision: {avg_precision:.4f}")
#         print(f"  Average Recall:    {avg_recall:.4f}")
#         print(f"  Average F1-Score:  {avg_f1:.4f}")
#         print(f"  Average Train Time:{avg_train_time:.2f}s")
#         print(f"  Average Eval Time: {avg_eval_time:.2f}s")

#         summary_stats[model_name] = {'avg_accuracy': avg_accuracy, 'avg_f1': avg_f1}

#     if summary_stats:
#          best_model_acc = max(summary_stats, key=lambda k: summary_stats[k]['avg_accuracy'])
#          best_model_f1 = max(summary_stats, key=lambda k: summary_stats[k]['avg_f1'])
#          print("\n" + "-"*40)
#          print(f"Best Model (Avg Accuracy): {best_model_acc} ({summary_stats[best_model_acc]['avg_accuracy']:.4f})")
#          print(f"Best Model (Avg F1-Score): {best_model_f1} ({summary_stats[best_model_f1]['avg_f1']:.4f})")
#          print("-" * 40)

#     print("\n--- Phase R3 Finished ---")



# -----------


# File: src/training/run_backtesting.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import os
import time
import traceback
import warnings

# Optional: Import advanced models if installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost library found.")
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost library not found, XGBoost model will be skipped.")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("LightGBM library found.")
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM library not found, LightGBM model will be skipped.")


# --- Configuration --- <<< REVIEW FOR 5min TEST >>> ---

# 1. Path to the enriched data file created in Phase R2
#    This should point to the PARQUET file generated by the
#    MODIFIED engineer_features.py script (using 5m yfinance data).
DATA_DIR = '../../data/'
# *** Ensure this filename matches the 5m output from engineer_features.py ***
DATA_FILENAME = 'eurusd_enriched_unknown.parquet' # <<< TARGETING  FILE
ENRICHED_DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

# 2. Target column name
TARGET_COL = 'target'

# 3. Features to EXCLUDE (if any).
EXCLUDE_COLS = ['open', 'high', 'low', 'close'] # Keep excluding raw prices

# 4. TimeSeriesSplit Configuration
N_SPLITS = 5
# MAX_TRAIN_SIZE = None
# GAP_DAYS = 0 # Gap might be more relevant in terms of rows/intervals for intraday

# 5. Models to evaluate
MODELS_TO_RUN = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'lightgbm'
]

# 6. Directory to save best models/results (optional)
RESULTS_DIR = './backtesting_results_1m' # Use a different results dir

# --- End Configuration ---


def load_data(filepath):
    """Loads the enriched data from Parquet."""
    print(f"\nLoading enriched data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: Enriched data file not found at '{filepath}'.")
        print("Please run the *modified* Phase R2 feature engineering script (for 5min data) first.")
        return None
    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load Parquet file: {e}")
        print(traceback.format_exc())
        return None

def get_model(model_name):
    """Returns an instance of the specified model."""
    if model_name == 'logistic_regression':
        return LogisticRegression(random_state=42, class_weight='balanced', max_iter=1500, solver='liblinear')
    elif model_name == 'random_forest':
        # May need different hyperparameters for higher frequency data
        return RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=15, n_jobs=-1, min_samples_leaf=5) # Adjusted params slightly
    elif model_name == 'gradient_boosting':
         # May need different hyperparameters for higher frequency data
        return GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.7) # Adjusted params slightly
    elif model_name == 'xgboost' and XGB_AVAILABLE:
        return xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, max_depth=7, learning_rate=0.1) # Added basic params
    elif model_name == 'lightgbm' and LGBM_AVAILABLE:
        return lgb.LGBMClassifier(random_state=42, objective='binary', metric='binary_logloss', n_estimators=100, max_depth=7, learning_rate=0.1) # Added basic params
    else:
        return None

def run_backtest_fold(model_name, X_train, y_train, X_test, y_test, fold_num):
    """Trains and evaluates a model on a single time series split."""
    fold_results = {}
    model = get_model(model_name)
    if model is None:
        print(f"Skipping model '{model_name}' (not available or unknown).")
        return None

    print(f"  Training {model_name}...")
    start_train_time = time.time()

    # Scale features for this fold
    scaler = StandardScaler()
    # Important: Ensure X_train/X_test are numeric before scaling
    X_train_numeric = X_train.select_dtypes(include=np.number)
    X_test_numeric = X_test.select_dtypes(include=np.number)
    # Check if columns were dropped (should not happen if engineer_features is correct)
    if X_train_numeric.shape[1] != X_train.shape[1]:
        print("WARN: Non-numeric columns detected before scaling!")

    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         model.fit(X_train_scaled, y_train)

    train_time = time.time() - start_train_time
    print(f"  Training finished in {train_time:.2f}s.")

    # Evaluate
    print(f"  Evaluating {model_name}...")
    start_eval_time = time.time()
    try:
        y_pred = model.predict(X_test_scaled)
    except Exception as pred_e:
        print(f"ERROR during prediction for {model_name}: {pred_e}")
        return None
    eval_time = time.time() - start_eval_time

    # Calculate metrics
    fold_results['accuracy'] = accuracy_score(y_test, y_pred)
    fold_results['precision'] = precision_score(y_test, y_pred, zero_division=0)
    fold_results['recall'] = recall_score(y_test, y_pred, zero_division=0)
    fold_results['f1'] = f1_score(y_test, y_pred, zero_division=0)
    # Storing full report dict might consume memory, only store key metrics if needed
    # fold_results['report'] = classification_report(y_test, y_pred, target_names=['PUT (0)', 'CALL (1)'], zero_division=0, output_dict=True)
    fold_results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    fold_results['train_time'] = train_time
    fold_results['eval_time'] = eval_time

    print(f"  Evaluation finished in {eval_time:.2f}s. Accuracy: {fold_results['accuracy']:.4f}")

    # Optionally save model/scaler from the last fold for inspection
    if fold_num == N_SPLITS: # Example: save only the last fold's model
        os.makedirs(RESULTS_DIR, exist_ok=True)
        try:
            joblib.dump(model, os.path.join(RESULTS_DIR, f'{model_name}_lastfold_model.pkl'))
            joblib.dump(scaler, os.path.join(RESULTS_DIR, f'{model_name}_lastfold_scaler.pkl'))
            print(f"  Saved model/scaler for {model_name} (last fold).")
        except Exception as save_e:
            print(f"  WARN: Failed to save model/scaler for {model_name}: {save_e}")


    return fold_results


# --- Main Backtesting Execution ---
if __name__ == "__main__":
    print("="*30)
    print("--- Starting Phase R3: Model Selection & Backtesting (5min yfinance Test) ---")
    print("="*30)

    # 1. Load Data
    df = load_data(ENRICHED_DATA_PATH)
    if df is None:
        exit()

    # 2. Prepare Features (X) and Target (y)
    print("\nPreparing features and target...")
    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found in the DataFrame.")
        exit()
    y = df[TARGET_COL]
    features_to_use = [col for col in df.columns if col != TARGET_COL and col not in EXCLUDE_COLS]
    print(f"Using {len(features_to_use)} features:")
    # Ensure features exist before selection
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        print(f"ERROR: Features specified not found in loaded data: {missing_features}")
        exit()
    X = df[features_to_use]

    print(f"  Feature examples: {features_to_use[:3]} ... {features_to_use[-3:]}")


    # --- FIX: Drop remaining NaNs AFTER selecting X and y ---
    print("\nDropping rows with NaNs from features (e.g., rolling windows, slope)...")
    initial_rows = len(X)
    # Convert object columns that should be numeric first
    for col in X.select_dtypes(include='object').columns:
         print(f"WARN: Column '{col}' has object dtype, attempting conversion to numeric.")
         X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna()
    y = y.loc[X.index] # Align y to match the remaining rows in X
    print(f"Dropped {initial_rows - len(X)} rows containing NaNs.")
    print(f"Final shape for backtesting: X={X.shape}, y={y.shape}")

    if X.empty or y.empty:
        print("ERROR: DataFrame became empty after final NaN drop. Check feature calculations in engineer_features.py.")
        exit()

    # Final check for NaNs/Infs (should pass)
    print("\nPerforming final check for NaNs/Infs in features (X)...")
    nan_check = X.isnull().sum()
    # Check for actual infinity values, not just large floats
    inf_mask = np.isinf(X.select_dtypes(include=np.number))
    inf_check_series = inf_mask.sum()

    columns_with_nan = nan_check[nan_check > 0]
    columns_with_inf = inf_check_series[inf_check_series > 0]

    if not columns_with_nan.empty or not columns_with_inf.empty:
         print("ERROR: NaNs or Infs still detected after final dropna. Investigation needed.")
         if not columns_with_nan.empty: print("Columns with NaNs:\n", columns_with_nan)
         if not columns_with_inf.empty: print("Columns with Infs:\n", columns_with_inf)
         exit()
    else:
         print("Final check passed: No NaNs or Infs found in feature columns.")


    # 3. Setup TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"\nUsing TimeSeriesSplit with {N_SPLITS} splits.")

    # 4. Run Backtesting Loop for each model
    all_results = {}

    for model_name in MODELS_TO_RUN:
        if model_name == 'xgboost' and not XGB_AVAILABLE: continue
        if model_name == 'lightgbm' and not LGBM_AVAILABLE: continue

        print(f"\n--- Evaluating Model: {model_name} ---")
        model_fold_results = []
        fold_num = 0

        for train_index, test_index in tscv.split(X):
            fold_num += 1
            print(f"\nProcessing Fold {fold_num}/{N_SPLITS}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
            # print(f"  Train period: {X_train.index.min()} -> {X_train.index.max()}") # Optional verbose
            # print(f"  Test period:  {X_test.index.min()} -> {X_test.index.max()}") # Optional verbose

            if X_train.empty or X_test.empty:
                 print("WARN: Empty train or test set in this fold, skipping.")
                 continue

            fold_result = run_backtest_fold(model_name, X_train, y_train, X_test, y_test, fold_num)
            if fold_result:
                 model_fold_results.append(fold_result)

        all_results[model_name] = model_fold_results

    # 5. Aggregate and Print Results
    print("\n" + "="*40)
    print("--- Backtesting Summary (5min yfinance Test) ---")
    print("="*40)

    summary_stats = {}
    for model_name, fold_results_list in all_results.items():
        if not fold_results_list:
            print(f"\nModel: {model_name} - No results (skipped or failed all folds).")
            continue

        print(f"\nModel: {model_name} ({len(fold_results_list)} folds)")
        avg_accuracy = np.mean([res['accuracy'] for res in fold_results_list])
        avg_precision = np.mean([res['precision'] for res in fold_results_list])
        avg_recall = np.mean([res['recall'] for res in fold_results_list])
        avg_f1 = np.mean([res['f1'] for res in fold_results_list])
        avg_train_time = np.mean([res['train_time'] for res in fold_results_list])
        avg_eval_time = np.mean([res['eval_time'] for res in fold_results_list])

        print(f"  Average Accuracy:  {avg_accuracy:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall:    {avg_recall:.4f}")
        print(f"  Average F1-Score:  {avg_f1:.4f}")
        print(f"  Average Train Time:{avg_train_time:.2f}s")
        print(f"  Average Eval Time: {avg_eval_time:.2f}s")

        summary_stats[model_name] = {'avg_accuracy': avg_accuracy, 'avg_f1': avg_f1}

    if summary_stats:
         best_model_acc = max(summary_stats, key=lambda k: summary_stats[k]['avg_accuracy'])
         best_model_f1 = max(summary_stats, key=lambda k: summary_stats[k]['avg_f1'])
         print("\n" + "-"*40)
         print(f"Best Model (Avg Accuracy): {best_model_acc} ({summary_stats[best_model_acc]['avg_accuracy']:.4f})")
         print(f"Best Model (Avg F1-Score): {best_model_f1} ({summary_stats[best_model_f1]['avg_f1']:.4f})")
         print("-" * 40)

    # Optional: Save summary results
    # os.makedirs(RESULTS_DIR, exist_ok=True)
    # summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    # summary_df.to_csv(os.path.join(RESULTS_DIR, 'backtesting_summary_5m.csv'))
    # print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'backtesting_summary_5m.csv')}")

    print("\n--- Phase R3 (5min yfinance Test) Finished ---")