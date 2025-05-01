# File: src/data_acquisition/download_dukascopy.py
# <<< USING THE 'dukascopy' library's Downloader class >>>
from dukascopy import Downloader

try:
    from dukascopy import Downloader # Use the Downloader class
    from dukascopy.core import TimeFrame # Assuming TimeFrame comes from core
    DUKASCOPY_LIB_AVAILABLE = True
except ImportError:
    print("ERROR: Failed to import 'Downloader' from 'dukascopy'.")
    print("Please ensure the 'dukascopy' library is installed correctly (`pip install dukascopy`).")
    DUKASCOPY_LIB_AVAILABLE = False

import pandas as pd
from datetime import datetime, timedelta
import os
import time
import traceback
import numpy as np

# --- Configuration --- <<< EDIT THESE PARAMETERS >>> ---

# 1. Instrument (Format as expected by this library, likely 'EURUSD')
INSTRUMENT = 'EURUSD'

# 2. Timeframe (Check library docs, '1m', '5m' likely strings)
#    Using TimeFrame enum might be safer if available and works
TIMEFRAME = '5m' # String format often works

# 3. Date Range (Inclusive) - Passed as individual components
#    *** START WITH A SMALLER RANGE FOR TESTING (e.g., 1 month) ***
START_YEAR = 2023
START_MONTH = 1
START_DAY = 1
END_YEAR = 2023
END_MONTH = 1 # Just download January 2023
END_DAY = 31
#    ************************************************************

# 4. Output Directory (relative to project root)
OUTPUT_PARENT_DIR = '../../data/dukascopy_dl' # Use a distinct name

# 5. Output Format (Must be supported by Downloader, 'csv' is likely)
OUTPUT_FORMAT = 'csv'

# 6. Chunking (We implement this externally by looping)
CHUNK_BY = 'month' # Options: 'month', 'year', None

# 7. Retries are handled within the loop structure
RETRIES = 2
RETRY_DELAY_SECONDS = 15

# --- End Configuration ---


def download_dukascopy_chunk_dl(downloader: Downloader, instrument, timeframe,
                                start_y, start_m, start_d, end_y, end_m, end_d,
                                output_dir, format):
    """Downloads a specific date chunk using the Downloader class."""
    start_dt_str = f"{start_y:04d}-{start_m:02d}-{start_d:02d}"
    end_dt_str = f"{end_y:04d}-{end_m:02d}-{end_d:02d}"
    print(f"  Attempting download via Downloader: {start_dt_str} to {end_dt_str}")

    # Define output path based on the chunk range and format
    output_filename_base = f"{instrument}_{timeframe}_{start_y:04d}{start_m:02d}{start_d:02d}_{end_y:04d}{end_m:02d}{end_d:02d}"
    output_path = os.path.join(output_dir, f"{output_filename_base}.{format}")

    if os.path.exists(output_path):
        print(f"  Skipping: File already exists - {output_path}")
        return True # Assume success

    success = False
    for attempt in range(RETRIES + 1):
        try:
            # Call the get_data method as shown in your example
            downloader.get_data(
                instrument=instrument,
                start_year=start_y,
                start_month=start_m,
                start_day=start_d,
                end_year=end_y,
                end_month=end_m,
                end_day=end_d,
                timeframe=timeframe,
                format=format,
                path=output_dir, # Does it use path? Check docs. Might save automatically?
                filename=output_filename_base # Pass filename without extension?
            )
            # Note: We assume get_data saves the file directly based on path/filename args.
            # We need to check if the file was actually created.

            time.sleep(1) # Give OS a moment

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                 print(f"  Downloader command finished. File saved: {output_path}")
                 success = True
                 break # Exit retry loop on success
            else:
                 print(f"  WARN: Output file not found or empty after download attempt {attempt+1}.")
                 # Add specific checks for partial downloads if possible
                 if attempt < RETRIES:
                      print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                      time.sleep(RETRY_DELAY_SECONDS)
                      continue
                 else: # Max retries failed
                      success = False
                      break

        except Exception as e:
            print(f"  ERROR during Downloader download (Attempt {attempt + 1}/{RETRIES + 1}): {e}")
            # print(traceback.format_exc()) # Uncomment for full error details
            if attempt < RETRIES:
                 print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                 time.sleep(RETRY_DELAY_SECONDS)
            else:
                 print(f"  ERROR: Max retries reached for chunk.")
                 success = False
                 break
    return success


# --- Main Download Script ---
if __name__ == "__main__":
    if not DUKASCOPY_LIB_AVAILABLE:
        exit() # Exit if import failed

    print("="*30)
    print("--- Dukascopy Data Downloader (using Downloader Class) ---")
    print("="*30)

    # Prepare output directory structure
    output_dir_instrument = os.path.join(OUTPUT_PARENT_DIR, INSTRUMENT)
    output_dir_final = os.path.join(output_dir_instrument, TIMEFRAME)
    print(f"Ensuring output directory exists: {output_dir_final}")
    os.makedirs(output_dir_final, exist_ok=True)

    # Initialize downloader instance
    print("Initializing Dukascopy Downloader...")
    try:
        # Does the Downloader class require any arguments? Assume no for now.
        dl = Downloader()
        print("Downloader initialized.")
    except Exception as init_e:
        print(f"ERROR: Failed to initialize Downloader: {init_e}")
        exit()

    # Define overall date range
    start_date_main = datetime(START_YEAR, START_MONTH, START_DAY)
    end_date_main = datetime(END_YEAR, END_MONTH, END_DAY)

    print(f"\nStarting download for {INSTRUMENT} ({TIMEFRAME})")
    print(f"From: {start_date_main.strftime('%Y-%m-%d')} To: {end_date_main.strftime('%Y-%m-%d')}")
    print(f"Chunking by: {CHUNK_BY}")
    print(f"Output format: {OUTPUT_FORMAT}")
    print(f"Saving {OUTPUT_FORMAT} files to: {output_dir_final}")
    print("-"*30)

    # Iterate through chunks
    current_start_dt = start_date_main
    total_chunks = 0
    successful_chunks = 0

    while current_start_dt <= end_date_main:
        total_chunks += 1
        # Determine end date for the current chunk
        if CHUNK_BY == 'month':
            next_month_start = (current_start_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
            current_end_dt = next_month_start - timedelta(days=1)
        elif CHUNK_BY == 'year':
            current_end_dt = current_start_dt.replace(month=12, day=31)
        else: # Fetch all at once (None)
            current_end_dt = end_date_main

        current_end_dt = min(current_end_dt, end_date_main) # Don't exceed overall end date

        print(f"\nProcessing Chunk {total_chunks}: {current_start_dt.strftime('%Y-%m-%d')} -> {current_end_dt.strftime('%Y-%m-%d')}")

        # Download this chunk using the Downloader instance
        success = download_dukascopy_chunk_dl(
            downloader=dl, # Pass the instance
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            start_y=current_start_dt.year,
            start_m=current_start_dt.month,
            start_d=current_start_dt.day,
            end_y=current_end_dt.year,
            end_m=current_end_dt.month,
            end_d=current_end_dt.day,
            output_dir=output_dir_final,
            format=OUTPUT_FORMAT
        )

        if success: successful_chunks += 1
        else: print(f"WARN: Chunk {total_chunks} may have failed after retries.")

        # Move to the start of the next chunk
        if CHUNK_BY is None: break
        else: current_start_dt = current_end_dt + timedelta(days=1)


    print("\n" + "="*30)
    print("--- Download Summary ---")
    print(f"Total Chunks Processed: {total_chunks}")
    print(f"Chunks Assumed Successful: {successful_chunks}")
    print(f"Failed Chunks (after retries): {total_chunks - successful_chunks}")
    print(f"Data saved in directory: {output_dir_final}")
    print("\nNOTE: Check the output directory to confirm files were downloaded.")
    print("You may need to merge these chunked files before feature engineering.")
    print("="*30)