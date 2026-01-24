import os
import pandas as pd
import numpy as np
from data_preprocessing import create_sliding_windows
# 引入之前的 create_sliding_windows 函数 (假设上面已经定义了)

# 定义动作对应的文件夹或文件名关键词
# 格式: '关键词': Label_ID
LABEL_MAP = {
    'rest': 0,   # 文件名包含 rest 的，标签为 0 (走路、喝水、乱动都命名为 rest_xxx.csv)
    'squat': 1,  # 文件名包含 squat 的，标签为 1
    'jump': 2    # 文件名包含 jump 的，标签为 2
}

def load_and_process_all_files(data_dir):
    all_X = []
    all_y = []
    
    # 遍历数据文件夹里的所有 CSV
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(data_dir, filename)
        
        # 自动匹配标签
        current_label = -1
        for keyword, label_id in LABEL_MAP.items():
            if keyword in filename.lower(): # 不区分大小写
                current_label = label_id
                break
        
        if current_label == -1:
            print(f"跳过文件 (未匹配到标签): {filename}")
            continue
            
        print(f"正在处理: {filename} -> Label {current_label}")
        
        try:
            df = pd.read_csv(file_path)
            # 调用之前的滑窗函数
            X_chunk, y_chunk = create_sliding_windows(df, current_label)
            
            if len(X_chunk) > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                print(f"  - 生成样本: {len(X_chunk)}")
            else:
                print(f"  - 数据过短，未生成样本")
                
        except Exception as e:
            print(f"  - 处理出错: {e}")

    # 合并所有数据
    if len(all_X) > 0:
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        return X_final, y_final
    else:
        return np.array([]), np.array([])

# === 主程序 ===
if __name__ == "__main__":
    # 假设你的所有 csv 都扔在这个文件夹里
    # 比如: rest_walk.csv, rest_stand.csv, squat_fast.csv...
    DATA_DIR = '/Users/aziko/Documents/grp/Machine-Learning-for-Smart-Fitness-Pod/data/raw2'
    
    X_train, y_train = load_and_process_all_files(DATA_DIR)
    
    if len(X_train) > 0:
        print("=" * 30)
        print(f"处理完毕！")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        # 简单检查一下数据平衡性
        unique, counts = np.unique(y_train, return_counts=True)
        print("各类别样本分布:", dict(zip(unique, counts)))
        
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        print("已保存 .npy 文件")
    else:
        print("错误：没有生成任何数据，请检查文件夹路径和文件名。")
