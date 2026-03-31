# Machine-Learning-for-Smart-Fitness-Pod

This repository contains the Machine Learning module for the Smart Fitness Pod. By processing data from wearable IMU sensors, it utilizes a deep learning model (MiniResNet) to perform real-time classification of different exercise categories, providing a foundation for subsequent repetition counting.

## Features

- **Multi-Action Recognition**: Currently supports the identification of 5 states/movements:
  - `Rest` (0) - Resting or daily activities
  - `Squat` (1) - Squats
  - `Bicep Curl` (2) - Bicep curls
  - `Bench Press` (3) - Bench presses
  - `Run` (4) - Running
- **Lightweight Model Design**: Utilizes a `MiniResNet` architecture based on 1D-CNN and residual blocks, highly optimized for processing time-series sensor data.
- **Edge Device Deployment**: Automatically converts the trained model into the `TensorFlow Lite (.tflite)` format, facilitating deployment on microcontrollers (such as nRF52840) or mobile edge devices.
- **Sliding Window Processing**: Supports customizable window sizes (default 208) and step sizes (default 20), specifically tuned for a 104Hz sensor sampling rate.

## Dataset Structure

Sensor CSV data should be organized according to the following directory structure to ensure strict separation between different phases:

- `data/raw1/`: **Temporary Buffer**. Used for storing unprocessed, unassigned, or newly collected raw data.
- `data/raw2/`: **Training Data**. The core dataset used for training the model. The script `data_preprocessing2.py` will read files from this directory to generate features and labels. Note: Filenames must contain specific action keywords (e.g., `rest`, `squat`, `bicep`, `bench`, `run`).
- `data/raw3/`: **Testing Data**. Used for independent inference and evaluation after the model is trained. The script `test.py` calls files from this directory to verify the model's generalization capabilities on unseen data.

## Project Structure
```
Machine-Learning-for-Smart-Fitness-Pod/
├── data/
│   ├── raw1/                  # Temporary buffer
│   ├── raw2/                  # Training data directory
│   └── raw3/                  # Testing data directory
├── src/
│   ├── data_preprocessing.py  # Base preprocessing functions (e.g., sliding window)
│   ├── data_preprocessing2.py # Batch processes raw2 data to generate .npy files
│   ├── train.py               # Trains the MiniResNet model and exports to TFLite
│   └── test.py                # Reads raw3 data for inference and visualization
├── X_train.npy                # (Generated) Preprocessed feature matrix
├── y_train.npy                # (Generated) Preprocessed label array
├── miniresnet_model.keras     # (Generated) Trained Keras native model
├── miniresnet_model.tflite    # (Generated) Converted TFLite model
└── README.md
```
## Quick Start / Usage Guide

### 1. Data Preprocessing
Ensure your training data is properly named (containing action keywords) and placed in the `data/raw2/` directory. Run the preprocessing script:
bash
python src/data_preprocessing2.py

*Upon success, `X_train.npy` and `y_train.npy` will be generated in the root directory.*

### 2. Model Training
Use the generated `.npy` data to train the model. The script will automatically split the validation set and train the `MiniResNet`:
bash
python src/train.py

*After completion, it will output the accuracy and a confusion matrix. If the accuracy meets the threshold (>85%), it will automatically generate `miniresnet_model.keras` and `miniresnet_model.tflite`.*

### 3. Model Testing and Visualization
Select an independent test file from `data/raw3/` to verify the actual detection performance. You can modify the `TEST_CSV` path in `test.py` to point to different test files:
bash
python src/test.py
