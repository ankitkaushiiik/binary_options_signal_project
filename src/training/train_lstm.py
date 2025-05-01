# File: src/training/train_lstm.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Used for validation split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # To save scaler
import os
import time
import traceback
import warnings
import gc # Garbage collector

# --- Configuration --- <<< REVIEW THIS SECTION >>> ---

# 1. Path to the enriched data file (with lags)
DATA_DIR = '../../data/'
DATA_FILENAME = 'eurusd_enriched_daily_with_lags.parquet' # <<< Ensure this is correct
ENRICHED_DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

# 2. Target column name
TARGET_COL = 'target'

# 3. Features to EXCLUDE (if any)
EXCLUDE_COLS = ['open', 'high', 'low', 'close'] # Exclude raw prices

# 4. LSTM Sequence Parameters
LOOKBACK_PERIOD = 10 # Number of past time steps to use for prediction (e.g., use last 10 days)

# 5. Train/Test Split Parameters
TEST_SIZE = 0.2 # Proportion of data for testing (taken from the end)

# 6. LSTM Model Architecture Parameters
LSTM_UNITS = 50      # Number of neurons in the LSTM layer
DROPOUT_RATE = 0.2   # Dropout rate for regularization
DENSE_UNITS = 25     # Units in the intermediate Dense layer
OUTPUT_UNITS = 1     # Binary classification (0 or 1)
ACTIVATION_OUTPUT = 'sigmoid' # For binary classification

# 7. Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 50          # Max number of epochs
BATCH_SIZE = 32       # Number of samples per gradient update
VALIDATION_SPLIT = 0.1 # Use last 10% of training data for validation during training
EARLY_STOPPING_PATIENCE = 5 # Stop training if validation loss doesn't improve for N epochs
LR_REDUCTION_PATIENCE = 3  # Reduce learning rate if validation loss plateaus
LR_REDUCTION_FACTOR = 0.5  # Factor to reduce learning rate by

# 8. Output directory (optional, for saving model/scaler)
RESULTS_DIR = './lstm_results'
MODEL_FILENAME = 'lstm_model.h5' # Keras model save format
SCALER_FILENAME = 'lstm_scaler.pkl'

# --- End Configuration ---


def load_data(filepath):
    """Loads the enriched data from Parquet."""
    print(f"\nLoading enriched data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: '{filepath}'")
        return None
    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
        # Ensure target is integer
        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df
    except Exception as e:
        print(f"ERROR: Failed to load Parquet file: {e}")
        print(traceback.format_exc())
        return None

def create_sequences(features, target, lookback):
    """Transforms data into sequences for LSTM."""
    X_seq, y_seq = [], []
    print(f"Creating sequences with lookback={lookback}...")
    # Start index requires enough history for the first lookback window
    start_index = lookback
    # Iterate through the data starting from the point where a full lookback is possible
    for i in range(start_index, len(features)):
        # Extract the sequence of features ending at index i-1
        feature_sequence = features[i-lookback:i]
        # Extract the target value at index i
        target_value = target[i]
        X_seq.append(feature_sequence)
        y_seq.append(target_value)

    print(f"Sequence creation complete. Number of sequences: {len(X_seq)}")
    return np.array(X_seq), np.array(y_seq)


# --- Main LSTM Training Script ---
if __name__ == "__main__":
    print("="*30)
    print("--- Starting LSTM Model Training ---")
    print("="*30)

    # 1. Load Data
    df = load_data(ENRICHED_DATA_PATH)
    if df is None: exit()

    # 2. Prepare Features (X) and Target (y)
    print("\nPreparing features and target...")
    if TARGET_COL not in df.columns: print(f"ERROR: Target column '{TARGET_COL}' not found."); exit()
    y_full = df[TARGET_COL]
    features_to_use = [col for col in df.columns if col != TARGET_COL and col not in EXCLUDE_COLS]
    print(f"Using {len(features_to_use)} features.")
    X_full_df = df[features_to_use]

    # Drop initial NaNs that might exist (e.g., from rolling/lags in previous step)
    print("\nDropping initial rows with NaNs...")
    initial_rows = len(X_full_df)
    X_full_df = X_full_df.dropna()
    y_full = y_full.loc[X_full_df.index]
    print(f"Dropped {initial_rows - len(X_full_df)} rows.")
    print(f"Shape before scaling: X={X_full_df.shape}, y={y_full.shape}")
    if X_full_df.empty: print("ERROR: Data empty after NaN drop."); exit()

    # 3. Scale Features
    print("\nScaling features using MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_full_df)
    print(f"Features scaled. Shape: {X_scaled.shape}")

    # 4. Create Sequences
    # We pass the scaled numpy array and the corresponding target Series
    X_seq, y_seq = create_sequences(X_scaled, y_full.values, LOOKBACK_PERIOD)
    if len(X_seq) == 0: print("ERROR: No sequences created. Check data length and lookback period."); exit()
    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}") # Should be (samples, lookback, features)

    # 5. Train/Test Split (Chronological)
    print(f"\nSplitting data into train/test sets (Test size: {TEST_SIZE*100:.0f}%)...")
    split_index = int(len(X_seq) * (1 - TEST_SIZE))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]
    print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shapes:  X={X_test.shape}, y={y_test.shape}")
    if len(X_train) == 0 or len(X_test) == 0: print("ERROR: Train or test set is empty after split."); exit()

    # 6. Build LSTM Model
    print("\nBuilding LSTM model...")
    n_features = X_train.shape[2] # Get number of features from sequence data
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(LOOKBACK_PERIOD, n_features), return_sequences=True), # Return seq for potential stacking
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2), # Second LSTM layer (optional, simpler)
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation='relu'),
        Dense(OUTPUT_UNITS, activation=ACTIVATION_OUTPUT)
    ])
    model.summary() # Print model architecture

    # 7. Compile Model
    print("\nCompiling model...")
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 8. Define Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_REDUCTION_FACTOR, patience=LR_REDUCTION_PATIENCE, min_lr=1e-6, verbose=1)

    # 9. Train Model
    print("\nTraining model...")
    start_train_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, # Use part of training data for validation
        callbacks=[early_stopping, reduce_lr],
        shuffle=False, # Important for time series not to shuffle train data
        verbose=1 # Show progress bar per epoch
    )
    train_time = time.time() - start_train_time
    print(f"Training finished in {train_time:.2f}s.")

    # 10. Evaluate Model
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get detailed predictions and report
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten() # Convert probabilities to 0/1 predictions

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['PUT (0)', 'CALL (1)'], zero_division=0))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))

    # 11. Save Model and Scaler (Optional)
    print("\nSaving model and scaler...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(RESULTS_DIR, SCALER_FILENAME)
    try:
        model.save(model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"ERROR saving Keras model: {e}")
    try:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    except Exception as e:
        print(f"ERROR saving scaler: {e}")

    # 12. Clean up memory (useful for large models/data)
    del model, X_train, y_train, X_test, y_test, X_seq, y_seq, X_scaled, df
    gc.collect()

    print("\n--- LSTM Training and Evaluation Finished ---")