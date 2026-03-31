import pandas as pd
import numpy as np

# ==========================================
# 1. Basic Configuration
# ==========================================
SAMPLING_RATE = 104       # Sampling rate (Hz)
WINDOW_SECONDS = 2      # Window duration: 2 seconds
OVERLAP_RATIO = 0.5       # Overlap ratio: 50%

# Trim configuration
TRIM_SECONDS = 3.0        # Remove 3 seconds from both start and end
TRIM_SAMPLES = int(SAMPLING_RATE * TRIM_SECONDS) # 3 * 104 = 312 samples

# Compute sliding window parameters
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)  # 260
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO)) # 130

print(f"Config check: window_size={WINDOW_SIZE}, step_size={STEP_SIZE}")
print(f"Trim strategy: remove {TRIM_SAMPLES} samples from both ends ({TRIM_SECONDS}s)")
# ==========================================
# 2. Core: Sliding Window Function
# ==========================================
def create_sliding_windows(df, label_id):
    """
    Input: DataFrame containing 6-axis data
    Output: (X, y)
    Pipeline: extract -> filter -> trim ends -> sliding window
    """

    # 1. Extract 6-axis data
    features = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
                'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    data = df[features].values

    # 2. No filtering, use raw data directly
    data_filtered = data.astype(float)

    # 3. Remove start/end data
    total_len = len(data_filtered)

    # Ensure data is long enough to trim
    if total_len > (2 * TRIM_SAMPLES + WINDOW_SIZE):
        print(f"  [Before trim] data length: {total_len}")

        # Slice: remove first and last 312 samples
        data_filtered = data_filtered[TRIM_SAMPLES : -TRIM_SAMPLES]

        print(f"  [After trim] data length: {len(data_filtered)} (removed {TRIM_SECONDS}s from both ends)")
    else:
        print(f"  [Warning] Data too short ({total_len}); cannot apply {TRIM_SECONDS}s trim. Skipping this step.")

    # 4. Start sliding window loop
    X_windows = []
    y_labels = []

    for i in range(0, len(data_filtered) - WINDOW_SIZE, STEP_SIZE):
        window = data_filtered[i : i + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            X_windows.append(window)
            y_labels.append(label_id)

    return np.array(X_windows), np.array(y_labels)

