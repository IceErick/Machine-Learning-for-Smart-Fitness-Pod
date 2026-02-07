import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 准备数据
# ==========================================
print("正在加载数据...")
# 请确保路径正确
try:
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
except FileNotFoundError:
    print("找不到 X_train.npy 或 y_train.npy，请先运行文件2生成数据。")
    exit()

print(f"数据形状: {X.shape}") # (Samples, 260, 6)

# 动态获取类别数量
num_classes = len(np.unique(y))
print(f"检测到类别数量: {num_classes}")

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. 构建 MiniResNet 模型 (核心修改部分)
# ==========================================

def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    """
    定义一个残差块:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
      |_______________________________________^
    """
    # 第一层卷积
    x = layers.Conv1D(filters, kernel_size, padding='same', strides=stride)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 第二层卷积
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 残差连接 (Shortcut)
    # 如果维度不匹配（比如步长!=1 或者 滤波器数量变了），需要对输入做 1x1 卷积来调整形状
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # 相加
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_mini_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # === 初始处理层 ===
    # 先把特征升维，保留时间信息
    x = layers.Conv1D(16, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.MaxPooling1D(3, strides=2, padding='same')(x) # 可选：如果要进一步压缩

    # === 残差块堆叠 (Mini版) ===
    # Block 1: 保持特征图大小
    x = resnet_block(x, filters=16, stride=1)
    x = layers.Dropout(0.2)(x) # 加上Dropout防止过拟合

    # Block 2: 增加通道，压缩时间维度 (stride=2)
    x = resnet_block(x, filters=32, stride=2)
    x = layers.Dropout(0.2)(x)

    # Block 3: 再增加通道，再压缩
    x = resnet_block(x, filters=64, stride=2)
    x = layers.Dropout(0.2)(x)

    # === 输出层 ===
    x = layers.GlobalAveragePooling1D()(x) # 将 (Time, Features) 变为 (Features,)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name="MiniResNet")
    return model

# 实例化模型
input_shape = (X_train.shape[1], X_train.shape[2]) # (260, 6)
model = build_mini_resnet(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. 训练
# ==========================================
print("\n开始训练 MiniResNet...")
history = model.fit(
    X_train, y_train,
    epochs=20,          # ResNet通常可以训练更多轮而不退化，建议设为 20-30
    batch_size=32,      # 稍微调大一点 batch size
    validation_data=(X_test, y_test)
)

# ==========================================
# 4. 评估与转换
# ==========================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集准确率: {acc*100:.2f}%")

if acc > 0.85: # 设定一个保存门槛
    print("✅ 模型表现良好，开始保存...")
    
    # 1. 保存 Keras 原生模型
    model.save('miniresnet_model.keras')
    
    # 2. 转换为 TFLite (针对 Mac 优化的 Concrete Function 方式)
    print("正在转换为 TFLite...")
    try:
        # 定义输入签名，固定 Batch Size = 1 (适合单次推理)
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([1, input_shape[0], input_shape[1]], model.inputs[0].dtype)
        )

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        
        with open('miniresnet_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("🎉 TFLite 模型转换成功: miniresnet_model.tflite")
        
    except Exception as e:
        print(f"❌ 本地转换失败: {e}")
        print("请使用 Google Colab 并上传 .keras 文件进行转换。")
else:
    print("⚠️ 准确率未达到预期，不进行保存。")

# 画图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.legend()
plt.show()