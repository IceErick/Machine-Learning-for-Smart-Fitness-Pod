import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class_names = {
    0: "rest",
    1: "squat",
    2: "bicep",
    3: "bench",
    4: "run"
}

# ==========================================
# 1. Prepare data
# ==========================================
print("Loading data...")
try:
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
except FileNotFoundError:
    print("X_train.npy or y_train.npy not found. Please run file 2 first to generate data.")
    exit()

print(f"Data shape: {X.shape}")

# Dynamically get the number of classes
num_classes = len(np.unique(y))
print(f"Detected number of classes: {num_classes}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. Build MiniResNet model
# ==========================================

def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    """
    residual block:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    # First conv layer
    x = layers.Conv1D(filters, kernel_size, padding='same', strides=stride)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv layer
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # Add
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_mini_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # === Initial processing layer ===
    # Upsample features while preserving temporal information
    x = layers.Conv1D(16, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # === Stacked residual blocks (Mini version) ===
    # Block 1: keep feature map size
    x = resnet_block(x, filters=16, stride=1)
    x = layers.Dropout(0.2)(x) # Add Dropout to prevent overfitting

    # Block 2: increase channels, compress time dimension (stride=2)
    x = resnet_block(x, filters=32, stride=2)
    x = layers.Dropout(0.2)(x)

    # Block 3: further increase channels, further compress
    x = resnet_block(x, filters=64, stride=2)
    x = layers.Dropout(0.2)(x)

    # === Output layer ===
    x = layers.GlobalAveragePooling1D()(x) # Reduce (Time, Features) to (Features,)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name="MiniResNet")
    return model

# Instantiate the model
input_shape = (X_train.shape[1], X_train.shape[2]) # (260, 6)
model = build_mini_resnet(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. Training
# ==========================================
print("\nStarting MiniResNet training...")
history = model.fit(
    X_train, y_train,
    epochs=20,          
    batch_size=32,      
    validation_data=(X_test, y_test)
)

# ==========================================
# 4. Evaluation and conversion
# ==========================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=[name for _, name in sorted(class_names.items())]))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[name for _, name in sorted(class_names.items())],
            yticklabels=[name for _, name in sorted(class_names.items())])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

if acc > 0.85: # Set a saving threshold
    print("✅ Model performance is good, starting save...")

    # 1. Save Keras native model
    model.save('miniresnet_model.keras')

    # 2. Convert to TFLite
    print("Converting to TFLite...")
    try:
        # Define input signature, fix Batch Size = 1
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
        print("🎉 TFLite model conversion successful: miniresnet_model.tflite")

    except Exception as e:
        print(f"❌ Local conversion failed: {e}")
        print("Please use Google Colab and upload the .keras file for conversion.")
else:
    print("⚠️ Accuracy did not meet expectation; skipping save.")

# Plot training curves
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
