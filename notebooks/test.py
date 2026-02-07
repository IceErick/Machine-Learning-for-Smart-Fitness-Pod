import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data_preprocessing import butter_lowpass_filter  # 复用你之前的滤波函数

# ==========================================
# 1. 配置
# ==========================================
# 必须与训练时完全一致
WINDOW_SIZE = 208
STEP_SIZE = 20    # 推理时步长可以小一点（比如20），这样检测更灵敏，时间分辨率更高
SAMPLING_RATE = 104

# 类别映射 (根据你之前定义的 LABEL_MAP)
class_names = {
    0: "Rest (乱动/静止)",
    1: "Squat (深蹲)",
    2: "Jump (跳跃)"
}

# ==========================================
# 2. 推理专用预处理函数
# ==========================================
def preprocess_for_inference(df):
    """
    与训练时的区别：
    1. 不裁剪首尾（我们需要看完整的测试过程）
    2. 不需要 label
    3. 返回窗口数据 + 每个窗口对应的时间点（用于画图）
    """
    features = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 
                'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    # 1. 提取数据
    raw_data = df[features].values
    
    # 2. 滤波 (必须做！否则会有噪声干扰)
    filtered_data = butter_lowpass_filter(raw_data)
    
    # 3. 滑窗
    X_windows = []
    time_indices = [] # 记录每个窗口结束时的索引，方便画图定位
    
    # 从 0 开始滑，直到数据结束
    for i in range(0, len(filtered_data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = filtered_data[i : i + WINDOW_SIZE]
        X_windows.append(window)
        time_indices.append(i + WINDOW_SIZE) # 记录窗口结束点
        
    return np.array(X_windows), np.array(time_indices), filtered_data

# ==========================================
# 3. 核心测试逻辑
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
    
    # 预处理
    X_test, time_indices, full_signal = preprocess_for_inference(df)
    
    if len(X_test) == 0:
        print("数据太短，无法构成一个完整的窗口。")
        return

    print(f"生成了 {len(X_test)} 个测试窗口，开始推理...")
    
    # === 预测 ===
    # predictions 是一个概率矩阵，比如 [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]...]
    predictions = model.predict(X_test)
    
    # 取出最大概率对应的索引 (0, 1, 或 2)
    predicted_classes = np.argmax(predictions, axis=1)
    # 取出这个预测的可信度 (Confidence)
    confidences = np.max(predictions, axis=1)

    # ==========================================
    # 4. 结果可视化
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # --- 画原始信号 (只画 Acc Z轴，通常深蹲这个轴变化最明显) ---
    # 你也可以改成 Acc Y 或 X，看你的设备佩戴方向
    plt.plot(full_signal[:, 2], color='gray', alpha=0.5, label='Filtered Acc Z')
    
    # --- 在信号上标记预测结果 ---
    print("\n=== 检测结果详情 ===")
    
    # 为了避免打印太多，我们只打印状态变化的时刻
    last_pred = -1
    
    for i, (pred_class, conf, end_idx) in enumerate(zip(predicted_classes, confidences, time_indices)):
        start_idx = end_idx - WINDOW_SIZE
        
        # 只有当置信度 > 0.6 时才认为是有效预测，否则视为噪音
        if conf > 0.6:
            color = 'white'
            if pred_class == 1: color = 'red'    # 深蹲 = 红
            elif pred_class == 2: color = 'blue' # 跳跃 = 蓝
            elif pred_class == 0: color = 'green' # 休息 = 绿
            
            # 在图上画线段
            plt.axvspan(start_idx, end_idx, color=color, alpha=0.1)
            
            # 简单的控制台输出 (去重)
            if pred_class != last_pred:
                timestamp = end_idx / SAMPLING_RATE
                print(f"时间 {timestamp:.1f}s -> 动作切换为: {class_names[pred_class]} (置信度: {conf:.2f})")
                last_pred = pred_class

    plt.title(f"模型测试结果: {os.path.basename(csv_path)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Z")
    plt.legend(loc='upper right')
    
    # 创建自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Rest (0)'),
        Patch(facecolor='red', alpha=1, label='Squat (1)'),
        Patch(facecolor='blue', alpha=1, label='Jump (2)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.show()

# ==========================================
# 5. 运行
# ==========================================
if __name__ == "__main__":
    # 替换成你要测试的新 CSV 文件路径
    # 最好找一个没参与过训练的文件
    TEST_CSV = '/Users/aziko/Documents/grp/Machine-Learning-for-Smart-Fitness-Pod/data/仰卧起坐2___l.csv'
    
    # 模型路径
    MODEL_FILE = 'miniresnet_model.keras'
    
    test_new_csv(TEST_CSV, MODEL_FILE)