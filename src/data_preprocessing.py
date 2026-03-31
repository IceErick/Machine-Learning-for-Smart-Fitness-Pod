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

print(f"配置确认: 窗口大小={WINDOW_SIZE}, 步长={STEP_SIZE}")
print(f"裁剪策略: 首尾各去除 {TRIM_SAMPLES} 个样本 ({TRIM_SECONDS}秒)")
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
        print(f"  [裁剪前] 数据长度: {total_len}")

        # Slice: remove first and last 312 samples
        data_filtered = data_filtered[TRIM_SAMPLES : -TRIM_SAMPLES]

        print(f"  [裁剪后] 数据长度: {len(data_filtered)} (去除了首尾各 {TRIM_SECONDS}s)")
    else:
        print(f"  [警告] 数据过短 ({total_len})，无法执行 {TRIM_SECONDS}s 裁剪，跳过此步骤。")

    # 4. Start sliding window loop
    X_windows = []
    y_labels = []

    for i in range(0, len(data_filtered) - WINDOW_SIZE, STEP_SIZE):
        window = data_filtered[i : i + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            X_windows.append(window)
            y_labels.append(label_id)

    return np.array(X_windows), np.array(y_labels)

