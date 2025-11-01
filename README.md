# Machine-Learning-for-Smart-Fitness-Pod
This repo is used to cooperate with group member to carry on machine learning. The trained model is expected to classify different categories of exercise and count repetition for that. 

## Sample Sensor Data
- `data/xiao_nr52840_sense_imu_sample.csv` contains 10 seconds of synthetic IMU output that mimics the Seeed Studio XIAO nRF52840 Sense Plus (accelerometer in g, gyroscope in dps, magnetometer in uT).
- `data/xiao_nr52840_sense_imu_sample_filtered.csv` contains filtered data which are reached after applying low-pass data filtering.
- Sampling rate is 50 Hz with three motion phases (idle, rhythmic lift, complex rotation) to help test preprocessing and model pipelines.
- Timestamps are millisecond offsets; convert to seconds by dividing by 1000 if needed.
