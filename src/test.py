import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data_preprocessing import butter_lowpass_filter

# ==========================================
# 1. Configuration
# ==========================================
WINDOW_SIZE = 208
STEP_SIZE = 20
SAMPLING_RATE = 104

# Class mapping
class_names = {
    0: "Rest",
    1: "Squat",
    2: "Bicep",
    3: "Bench",
    4: "Run"
}

# ==========================================
# 2. Inference-specific preprocessing function
# ==========================================
def preprocess_for_inference(df):
    features = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
                'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    # 1. Extract data
    raw_data = df[features].values

    # 2. Filter
    filtered_data = butter_lowpass_filter(raw_data)

    # 3. Sliding window
    X_windows = []
    time_indices = [] # Record the end index of each window for plot positioning

    for i in range(0, len(filtered_data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = filtered_data[i : i + WINDOW_SIZE]
        X_windows.append(window)
        time_indices.append(i + WINDOW_SIZE) # Record window end point

    return np.array(X_windows), np.array(time_indices), filtered_data

# ==========================================
# 3. Core testing logic
# ==========================================
def test_new_csv(csv_path, model_path):
    print(f"正在加载模型: {model_path} ...")
    try:
        model = tf.keras.models.load_model(model_path)
    except OSError:
        print("错误：找不到模型文件。请确保 miniresnet_model.keras 存在。")
        return

    print(f"正在读取数据: {csv_path} ...")
    if not os.path.exists(csv_path):
        print("错误：找不到CSV文件。")
        return

    df = pd.read_csv(csv_path)

    # Preprocess
    X_test, time_indices, full_signal = preprocess_for_inference(df)

    if len(X_test) == 0:
        print("数据太短，无法构成一个完整的窗口。")
        return

    print(f"生成了 {len(X_test)} 个测试窗口，开始推理...")

    # === Predict ===
    # predictions is a probability matrix, e.g. [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]...]
    predictions = model.predict(X_test)

    # Get the index with highest probability (0, 1, or 2)
    predicted_classes = np.argmax(predictions, axis=1)
    # Get the confidence of the prediction
    confidences = np.max(predictions, axis=1)

    # ==========================================
    # 4. Result visualization
    # ==========================================
    plt.figure(figsize=(12, 6))

    # --- Plot raw signal (only Acc Z-axis, usually most prominent for squats) ---
    plt.plot(full_signal[:, 2], color='gray', alpha=0.5, label='Filtered Acc Z')

    # --- Mark prediction results on the signal ---
    print("\n=== 检测结果详情 ===")

    # To avoid too much output, only print when the predicted state changes
    last_pred = -1

    for i, (pred_class, conf, end_idx) in enumerate(zip(predicted_classes, confidences, time_indices)):
        start_idx = end_idx - WINDOW_SIZE

        # Only treat as valid prediction if confidence > 0.6, otherwise consider as noise
        if conf > 0.6:
            color = 'white'
            if pred_class == 1: color = 'red'    # Squat = red
            elif pred_class == 2: color = 'blue' # Bicep curl = blue
            elif pred_class == 0: color = 'green' # Rest = green
            elif pred_class == 3: color = 'orange' # Bench press = orange
            elif pred_class == 4: color = 'purple' # Run = purple


            # Draw segment on the plot
            plt.axvspan(start_idx, end_idx, color=color, alpha=0.1)

            # Simple console output
            if pred_class != last_pred:
                timestamp = end_idx / SAMPLING_RATE
                print(f"时间 {timestamp:.1f}s -> 动作切换为: {class_names[pred_class]} (置信度: {conf:.2f})")
                last_pred = pred_class

    plt.title(f"模型测试结果: {os.path.basename(csv_path)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Z")
    plt.legend(loc='upper right')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Rest (0)'),
        Patch(facecolor='red', alpha=1, label='Squat (1)'),
        Patch(facecolor='blue', alpha=1, label='Bicep Curl (2)'),
        Patch(facecolor='orange', alpha=1, label='Bench (3)'),
        Patch(facecolor='purple', alpha=1, label='Run (4)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.show()

# ==========================================
# 5. Run
# ==========================================
if __name__ == "__main__":
    # A file that was not used in training
    TEST_CSV = 'data/raw3/sensor_data_1766482134374_run周跑步9kph5分钟.csv'

    # Model path
    MODEL_FILE = 'miniresnet_model.keras'

    test_new_csv(TEST_CSV, MODEL_FILE)
