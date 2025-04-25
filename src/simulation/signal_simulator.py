# File: src/simulation/signal_simulator.py

import pandas as pd
import numpy as np
import joblib
import redis
import time
import json
import sys
import os
from datetime import datetime
import traceback

# --- Configuration ---
# Adjust path to find the 'src' directory for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    # Import data processing function from its location
    from data_processing.processor import fetch_and_process_data
    print("Successfully imported data processor.")
except ImportError as e:
     print(f"Error: Could not import 'fetch_and_process_data'. Ensure Phase 1 script is in src/data_processing/processor.py")
     print(f"PYTHONPATH: {sys.path}")
     print(f"Import Error: {e}")
     exit()

# --- Paths and Config ---
# IMPORTANT: Adjust path to where model/scaler were saved in Phase 2
ARTIFACTS_DIR = os.path.join(SRC_DIR, 'data_processing') # Points to where pkl files are currently located
MODEL_FILENAME = "simple_logistic_regression_model.pkl"
SCALER_FILENAME = "standard_scaler.pkl"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(ARTIFACTS_DIR, SCALER_FILENAME)
USE_SCALER = os.path.exists(SCALER_PATH) # Auto-detect scaler

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_CHANNEL = 'signals_channel' # Channel to publish on

# Data source for simulation (use recent data, maybe a different period than training)
TICKER = "EURUSD=X"
# Use automatic period fetching to get recent data
SIM_START_DATE = None
SIM_END_DATE = None
SIM_INTERVAL = "5m" # Must match interval model was trained on

SIMULATION_SPEED_FACTOR = 0.1 # Lower value = faster simulation (e.g., 0.1 -> 10x speed). Set > 0.
MIN_CONFIDENCE_THRESHOLD = 0.0 # Minimum confidence (%) to publish signal (0.0 = publish all)

# Features used by the loaded model (MUST MATCH Phase 2 FEATURE_COLS)
FEATURE_COLS = [
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'RSI_14',
    'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0'
]

def load_model_and_scaler(model_path, scaler_path, use_scaler):
    """Loads the trained model and scaler."""
    print(f"INFO: Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None, None
    try:
        model = joblib.load(model_path)
        print("INFO: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}. Error: {e}")
        return None, None

    scaler = None
    if use_scaler:
        print(f"INFO: Loading scaler from {scaler_path}...")
        if not os.path.exists(scaler_path):
            print(f"WARN: Scaler file specified but not found at {scaler_path}. Proceeding without scaling.")
            use_scaler = False # Disable scaling if file not found
        else:
            try:
                scaler = joblib.load(scaler_path)
                print("INFO: Scaler loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load scaler from {scaler_path}. Error: {e}")
                return None, None # Treat scaler load failure as critical? Or proceed without scaling? Let's exit.
    else:
        print("INFO: Scaler not used.")

    return model, scaler

# --- Main Simulation Loop ---
if __name__ == "__main__":
    print("="*30)
    print("--- Starting Phase 3: Signal Simulation Script ---")
    print("="*30)

    # 1. Connect to Redis Publisher
    redis_publisher = None
    while redis_publisher is None:
         try:
             print(f"INFO: Attempting to connect to Redis Publisher at {REDIS_HOST}:{REDIS_PORT}")
             # decode_responses=True means Redis returns strings, not bytes
             redis_publisher = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
             redis_publisher.ping()
             print(f"INFO: Connected to Redis Publisher successfully.")
         except redis.exceptions.ConnectionError as e:
             print(f"WARN: Could not connect to Redis Publisher. Retrying in 5 seconds... Error: {e}")
             time.sleep(5)
         except Exception as e:
              print(f"FATAL: Unexpected error connecting to Redis: {e}")
              exit()


    # 2. Load Model and Scaler
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH, USE_SCALER)
    if model is None:
         print("ERROR: Model could not be loaded. Exiting.")
         exit()
    if USE_SCALER and scaler is None:
         print("ERROR: Scaler was intended to be used but could not be loaded. Exiting.")
         exit()


    # 3. Fetch Simulation Data
    print("\nINFO: Fetching data for simulation...")
    sim_data = fetch_and_process_data(ticker=TICKER, start_date=SIM_START_DATE, end_date=SIM_END_DATE, interval=SIM_INTERVAL)

    if sim_data is None or sim_data.empty:
        print("ERROR: Failed to get simulation data. Exiting.")
        exit()

    # Ensure all required features are present in the fetched data
    missing_cols = [col for col in FEATURE_COLS if col not in sim_data.columns]
    if missing_cols:
         print(f"ERROR: Missing required feature columns in simulation data: {missing_cols}")
         print(f"Available columns: {sim_data.columns.tolist()}")
         exit()
    print(f"INFO: Simulation data loaded. Shape: {sim_data.shape}")

    # Select only the feature columns needed for prediction
    sim_features_df = sim_data[FEATURE_COLS]

    # 4. Start Simulation Loop
    print("\n--- Starting simulation loop (Press Ctrl+C to stop) ---")
    total_signals_generated = 0
    total_signals_published = 0
    interval_seconds = pd.Timedelta(SIM_INTERVAL).total_seconds()
    sleep_duration = max(0.01, interval_seconds * SIMULATION_SPEED_FACTOR) # Ensure sleep > 0
    print(f"INFO: Simulating 1 interval ({SIM_INTERVAL} = {interval_seconds}s) every {sleep_duration:.2f} real seconds.")

    for index, row in sim_features_df.iterrows():
        try:
            current_timestamp_iso = index.isoformat() # Get timestamp from DataFrame index
            features_for_prediction = row.values.reshape(1, -1) # Reshape for model prediction (1 sample, N features)

            # Scale features if scaler is loaded
            if scaler:
                features_for_prediction = scaler.transform(features_for_prediction)

            # Make prediction (0 or 1)
            prediction = model.predict(features_for_prediction)[0]
            # Get prediction probabilities P(class=0), P(class=1)
            probabilities = model.predict_proba(features_for_prediction)[0]

            total_signals_generated += 1
            signal_type = None
            confidence = 0.0

            if prediction == 1: # Model predicts CALL (Up)
                signal_type = "CALL"
                confidence = probabilities[1] * 100 # Confidence is probability of predicted class '1'
            elif prediction == 0: # Model predicts PUT (Down)
                signal_type = "PUT"
                confidence = probabilities[0] * 100 # Confidence is probability of predicted class '0'

            # --- Publish Signal if confidence meets threshold ---
            if signal_type is not None and confidence >= MIN_CONFIDENCE_THRESHOLD:
                signal_payload = {
                    "timestamp": current_timestamp_iso, # Use ISO format timestamp
                    "asset": TICKER.replace("=X", ""), # Clean up ticker name
                    "type": signal_type,
                    "expiry_minutes": int(interval_seconds / 60),
                    "confidence": round(confidence, 2),
                    "interval": SIM_INTERVAL,
                    # "raw_prediction": int(prediction), # Optional: include raw prediction
                    # "model_version": "baseline_lr_v1" # Optional: add model info
                }
                message = json.dumps(signal_payload) # Convert dict to JSON string

                # Publish to Redis channel
                try:
                    redis_publisher.publish(REDIS_CHANNEL, message)
                    print(f"Published -> Timestamp: {current_timestamp_iso}, Asset: {signal_payload['asset']}, Type: {signal_type}, Conf: {confidence:.2f}%")
                    total_signals_published += 1
                except redis.exceptions.ConnectionError as e:
                     print(f"WARN: Redis connection error during publish: {e}. Skipping publish.")
                     # Consider attempting reconnect or exiting after repeated failures
                except Exception as e:
                     print(f"WARN: Unexpected error during Redis publish: {e}")


            # Simulate time passing between intervals
            time.sleep(sleep_duration)

        except KeyboardInterrupt:
            print("\nINFO: Simulation stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"ERROR: Error during simulation loop at index {index}: {e}")
            print(traceback.format_exc())
            print("WARN: Skipping this interval and continuing...")
            time.sleep(1) # Pause briefly on error

    print("\n" + "="*30)
    print("--- Simulation Finished ---")
    print(f"Total intervals processed: {total_signals_generated}")
    print(f"Total signals published (Confidence >= {MIN_CONFIDENCE_THRESHOLD}%): {total_signals_published}")
    print("="*30)