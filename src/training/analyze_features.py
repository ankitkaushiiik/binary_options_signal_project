# File: src/training/analyze_features.py

import pandas as pd
import numpy as np
import lightgbm as lgb 
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
import traceback
import matplotlib.pyplot as plt # For plotting
import warnings  # Add warnings import

# --- Configuration ---
DATA_DIR = '../../data/'
# Use the enriched intraday file
DATA_FILENAME = 'eurusd_enriched_unknown.parquet' 
ENRICHED_DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)
TARGET_COL = 'target'
# Exclude the same columns as in backtesting
EXCLUDE_COLS = ['open', 'high', 'low', 'close']
# Number of top features to display/save
N_TOP_FEATURES = 30
# Output directory for plots/results
RESULTS_DIR = './feature_analysis_results'
# --- End Configuration ---

def load_data(filepath):
    """Loads the enriched data from Parquet."""
    print(f"Loading enriched data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: Enriched data file not found at '{filepath}'.")
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

# --- Main Analysis Script ---
if __name__ == "__main__":
    print("="*30)
    print("--- Starting Feature Importance Analysis ---")
    print("="*30)

    # 1. Load Data
    df = load_data(ENRICHED_DATA_PATH)
    if df is None:
        exit()

    # 2. Prepare Features (X) and Target (y)
    print("\nPreparing features and target...")
    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found.")
        exit()
    y = df[TARGET_COL]
    features_to_use = [col for col in df.columns if col != TARGET_COL and col not in EXCLUDE_COLS]
    print(f"Using {len(features_to_use)} features.")
    X = df[features_to_use]

    # 3. Drop NaNs (Consistent with backtesting script)
    print("\nDropping rows with NaNs from features...")
    initial_rows = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    print(f"Dropped {initial_rows - len(X)} rows.")
    print(f"Final shape for analysis: X={X.shape}, y={y.shape}")

    if X.empty or y.empty:
        print("ERROR: DataFrame empty after final NaN drop.")
        exit()

    # 4. Scale Features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns) # Keep column names

    # 5. Train LightGBM Model on Full Dataset
    print("\nTraining LightGBM model on full dataset...")
    start_train_time = time.time()
    # Use similar parameters as backtesting, potentially slightly more estimators
    lgbm_model = lgb.LGBMClassifier(random_state=42, objective='binary', metric='binary_logloss', n_estimators=150, max_depth=10, learning_rate=0.1, class_weight='balanced', n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm_model.fit(X_scaled, y)

    train_time = time.time() - start_train_time
    print(f"Training finished in {train_time:.2f}s.")

    # 6. Extract Feature Importances
    print("\nExtracting feature importances...")
    importances = lgbm_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # 7. Display and Save Top Features
    print(f"\n--- Top {N_TOP_FEATURES} Features ---")
    print(feature_importance_df.head(N_TOP_FEATURES))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    importance_csv_path = os.path.join(RESULTS_DIR, 'feature_importances.csv')
    try:
        feature_importance_df.to_csv(importance_csv_path, index=False)
        print(f"\nFull feature importances saved to: {importance_csv_path}")
    except Exception as e:
        print(f"ERROR saving feature importances: {e}")

    # 8. Plot Top Features (Optional but recommended)
    print("\nGenerating feature importance plot...")
    try:
        plt.figure(figsize=(10, N_TOP_FEATURES / 2)) # Adjust figure size
        plt.barh(feature_importance_df['feature'][:N_TOP_FEATURES][::-1], # Plot top N, reverse for plot order
                feature_importance_df['importance'][:N_TOP_FEATURES][::-1])
        plt.xlabel('Importance (LGBM Gain)')
        plt.ylabel('Feature')
        plt.title(f'Top {N_TOP_FEATURES} Feature Importances')
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, 'top_feature_importances.png')
        plt.savefig(plot_path)
        print(f"Feature importance plot saved to: {plot_path}")
        # plt.show() # Uncomment to display plot immediately if running locally
    except Exception as e:
        print(f"ERROR generating plot: {e}")
        print("Ensure matplotlib is installed: pip install matplotlib")

    print("\n--- Feature Analysis Finished ---")