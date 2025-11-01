# Machine-Learning-for-Smart-Fitness-Pod
This repo is used to cooperate with group member to carry on machine learning. The trained model is expected to classify different categories of exercise and count repetition for that. 

## About This Project: Smart Fitness Pod - Machine Learning Module

Real-time exercise classification and repetition counting using IMU data from wearable sensors.

### Features

- **Real-time Exercise Classification**: Identify different workout movements
- **Repetition Counting**: Automatically count exercise repetitions
- **TensorFlow Lite Support**: Optimized for edge device deployment
- **IMU Data Processing**: Handle accelerometer and gyroscope data

### Installation

```bash
pip install -e .
```

### Usage

```python
from fitness_ai import ExerciseClassifier

# Initialize classifier
classifier = ExerciseClassifier()

# Classify exercise from IMU data
result = classifier.predict(imu_data)
```

### Project Structure

```
fitness_ai/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── preprocessor.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── exercise_classifier.py
│   └── repetition_counter.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── evaluator.py
└── utils/
    ├── __init__.py
    ├── feature_extractor.py
    └── model_converter.py
```



## Progress right now (still updating)

### Sample Sensor Data
- `data/xiao_nr52840_sense_imu_sample.csv` contains 10 seconds of synthetic IMU output that mimics the Seeed Studio XIAO nRF52840 Sense Plus (accelerometer in g, gyroscope in dps, magnetometer in uT).
- `data/xiao_nr52840_sense_imu_sample_filtered.csv` contains filtered data which are reached after applying low-pass data filtering.
- Sampling rate is 50 Hz with three motion phases (idle, rhythmic lift, complex rotation) to help test preprocessing and model pipelines.
- Timestamps are millisecond offsets; convert to seconds by dividing by 1000 if needed.