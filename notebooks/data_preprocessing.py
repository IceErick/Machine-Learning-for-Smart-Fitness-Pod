import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

# ==========================================
# 1. 基础配置
# ==========================================
SAMPLING_RATE = 104       # 采样率 (Hz)
WINDOW_SECONDS = 2      # 窗口时长: 2.5秒
OVERLAP_RATIO = 0.5       # 重叠率: 50%

# 裁剪配置 (新增)
TRIM_SECONDS = 3.0        # 首尾各去除 3 秒
TRIM_SAMPLES = int(SAMPLING_RATE * TRIM_SECONDS) # 3 * 104 = 312 个点

# 计算滑窗参数
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)  # 260
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO)) # 130

print(f"配置确认: 窗口大小={WINDOW_SIZE}, 步长={STEP_SIZE}")
print(f"裁剪策略: 首尾各去除 {TRIM_SAMPLES} 个样本 ({TRIM_SECONDS}秒)")

# ==========================================
# 2. 滤波器函数
# ==========================================
def butter_lowpass_filter(data, cutoff=5.0, fs=104.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0) 
    return y

# ==========================================
# 3. 核心：滑窗切分函数 (已修改)
# ==========================================
def create_sliding_windows(df, label_id):
    """
    输入: 包含 6 轴数据的 DataFrame
    输出: (X, y)
    流程: 提取 -> 滤波 -> 裁剪首尾 -> 滑窗
    """
    
    # 1. 提取 6 轴数据
    features = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 
                'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    data = df[features].values
    
    # 2. 先滤波！(建议先滤波再裁剪，保证信号边缘平滑)
    data_filtered = butter_lowpass_filter(data)
    
    # 3. === 新增：去除首尾数据 ===
    total_len = len(data_filtered)
    
    # 确保数据够长，别剪没了
    if total_len > (2 * TRIM_SAMPLES + WINDOW_SIZE):
        print(f"  [裁剪前] 数据长度: {total_len}")
        
        # 切片操作：去掉前312个，去掉后312个
        # 注意：Python切片是 [start : end]，负数索引代表倒数
        data_filtered = data_filtered[TRIM_SAMPLES : -TRIM_SAMPLES]
        
        print(f"  [裁剪后] 数据长度: {len(data_filtered)} (去除了首尾各 {TRIM_SECONDS}s)")
    else:
        print(f"  [警告] 数据过短 ({total_len})，无法执行 {TRIM_SECONDS}s 裁剪，跳过此步骤。")

    # 4. 开始滑窗循环
    X_windows = []
    y_labels = []
    
    for i in range(0, len(data_filtered) - WINDOW_SIZE, STEP_SIZE):
        window = data_filtered[i : i + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            X_windows.append(window)
            y_labels.append(label_id)
            
    return np.array(X_windows), np.array(y_labels)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 使用绝对路径或相对路径
    # 请确保这个文件路径是存在的
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设你的数据文件在 data/raw 目录下，或者直接在脚本同级目录
    # 这里示例直接写文件名，你需要根据实际情况修改
    csv_filename = '/Users/aziko/Documents/grp/Machine-Learning-for-Smart-Fitness-Pod/data/raw/2025-12-31T16:07:47.300271_athlete4_squat_25.csv'
    
    try:
        df = pd.read_csv(csv_filename)
        
        # 假设深蹲的 Label ID 是 1
        SQUAT_LABEL = 1
        
        # 执行切分
        X, y = create_sliding_windows(df, SQUAT_LABEL)
        
        print("-" * 30)
        print("处理完成！")
        print(f"生成的样本数量 (X shape): {X.shape}")
        print(f"生成的标签数量 (y shape): {y.shape}")
        
        if len(X) > 0:
            print(f"单个样本形状: {X[0].shape}")
            # 保存
            np.save('X_train.npy', X)
            np.save('y_train.npy', y)
            print("已保存 X_train.npy 和 y_train.npy")
        else:
            print("警告：没有生成任何样本，可能是数据裁剪后太短了。")
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_filename}")
        print("请检查路径是否正确，或者使用绝对路径。")