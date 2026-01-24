import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. 准备数据
# ==========================================
# 假设你已经保存了这两组数据
# 如果你用之前的脚本一次性读取了文件夹，直接 load X_train.npy 即可，跳过合并步骤

    # 场景 A: 你只有一个大文件 (推荐)
print("尝试加载完整数据集...")
X = np.load('/Users/aziko/Documents/grp/Machine-Learning-for-Smart-Fitness-Pod/X_train.npy')
y = np.load('/Users/aziko/Documents/grp/Machine-Learning-for-Smart-Fitness-Pod/y_train.npy')


print("-" * 30)
print(f"数据加载完毕！")
print(f"总样本数: {len(X)}")
print(f"  - 乱动 (Label 0): {np.sum(y == 0)}")
print(f"  - 深蹲 (Label 1): {np.sum(y == 1)}")
print("-" * 30)

# ==========================================
# 2. 切分训练集和测试集
# ==========================================
# test_size=0.2 表示留 20% 考试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. 搭建模型 (二分类)
# ==========================================
model = models.Sequential([
    # 输入层: (260, 6)
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 卷积层
    layers.Conv1D(16, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    
    # 展平与全连接
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    # 输出层: 2个神经元 (0和1)
    layers.Dense(2, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. 训练
# ==========================================
print("开始训练...")
history = model.fit(
    X_train, y_train,
    epochs=15,            # 跑15轮就够了
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ==========================================
# 5. 结果分析
# ==========================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集准确率: {acc*100:.2f}%")

if acc > 0.9:
    print("✅ 模型表现优秀！")
    
    # === 第一步：务必先保存 Keras 原生模型 (救命稻草) ===
    # 这样即使下面转换崩了，你也不用重新训练
    model_save_path = 'exercise_model_saved.keras'
    model.save(model_save_path)
    print(f"💾 已保存原生模型到: {model_save_path} (如果下面转换崩溃，请用这个文件去 Google Colab 转换)")

    print("\n正在转换模型为 TFLite...")

    # === 第二步：尝试 Mac 兼容性更好的转换方式 ===
    # Mac M1/M2 经常在 TFLite 转换时崩溃，我们尝试禁用一些优化
    try:
        # 1. 定义具体的输入签名 (Concrete Function)
        # 注意：这里需要明确指定 Batch Size 为 1，这通常能解决 LLVM 推断错误
        # 你的 X_train.shape[1] 是时间步 (260)，[2] 是特征数 (6)
        input_shape = (1, X_train.shape[1], X_train.shape[2])
        
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(input_shape, model.inputs[0].dtype)
        )

        # 2. 使用 from_concrete_functions 替代 from_keras_model
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # 3. 设置算子支持
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS 
        ]
        
        tflite_model = converter.convert()
        
        # 4. 保存
        save_path = 'exercise_model.tflite'
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"🎉 恭喜！通过 Concrete Function 方法转换成功！已保存为 '{save_path}'")

    except Exception as e:
        print(f"❌ 本地转换依然失败: {e}")
        print("💡 请务必使用方案一（Google Colab）进行转换。")
else:
    print("⚠️ 准确率有点低，可能需要检查数据质量或调整模型。")
# 画图
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend()
plt.show()