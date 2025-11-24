# Smart Fitness Pod — Machine Learning Module

End‑to‑end workflow for collecting IMU data from the Seeed Studio XIAO nRF52840 Sense Plus, preprocessing it, segmenting repetitions, and preparing data for model training and on‑device deployment.

## Repository Layout
- `src/fitness_ai/` — Python package
  - `utils/paths.py` — project/data path helpers (`raw_data_path`, `processed_data_path`, `segmented_data_path`, `get_metadata_index_path`)
  - `data/` — Phase 1 utilities
    - `constants.py` — sampling rate (104 Hz), units (m/s^2, rad/s), window/segment defaults
    - `metadata.py` — filename parsing (`athlete_<id>_<exercise>_<datetime>_<reps>.csv`) and metadata index helpers
    - `preprocess.py` — load, resample/interpolate, low‑pass filter (4th‑order 10 Hz), add `time_s`, write filtered CSVs
    - `segmentation.py` — sliding windows and peak/valley‑based repetition slicing
- `notebooks/` — exploratory workflows (e.g., LowPassFilter, SegmentationPeakDetection) that now import `fitness_ai` paths/utilities
- `data/` — expected layout
  - `raw/<exercise>/athlete_*_*.csv`
  - `filtered/<exercise>/...` (preprocessed)
  - `segmented/<exercise>/...` (per‑rep slices)
  - `metadata.csv` (optional index written by helpers)
- `environment.yml` — conda environment spec

## Setup
```bash
conda env create -f environment.yml
conda activate Machine-Learning-for-Smart-Fitness-Pod
pip install -e .
```

## Data Naming & Format (Phase 1)
- Filename: `athlete_<id>_<exercise>_<datetime>_<reps>.csv` (e.g., `athlete_01_deep-squat_2025-11-24T17-06-12_20.csv`)
- Columns: `Timestamp, Acceleration_X/Y/Z, Gyro_X/Y/Z`
- Sampling: 104 Hz; acceleration in m/s^2; angular velocity in rad/s
- Storage: raw under `data/raw/<exercise>/`; filtered under `data/filtered/<exercise>/`; segmented under `data/segmented/<exercise>/`

## Workflow
1) **Collect & log metadata**
   - Place sensor on wrist/ankle/equipment; record placement, firmware, orientation notes.
   - Keep a clap/button mark for video/time sync; ensure 5 s stationary for quick bias check.
   - Save files with the naming rule above.
2) **Preprocess (Phase 1)**
   - Resample to 104 Hz with interpolation for small gaps.
   - Add `time_s` (seconds from start).
   - Apply 4th‑order 10 Hz Butterworth low‑pass (zero‑phase) to accel/gyro.
   - Write to `data/filtered/<exercise>/<same filename>`.
3) **Segmentation**
   - Use sliding windows (2.0 s, 0.5 s step) for fixed‑length samples, or
   - Detect peaks/valleys (min interval ~1.0 s, prominence ~3×std) on a dominant axis (e.g., Accel_Z) and slice with pre/post windows (0.8 s / 1.0 s).
   - Save per‑rep slices to `data/segmented/<exercise>/`.
4) **Feature engineering & training (future phases)**
   - Extract time/frequency features or use sequence models.
   - Split train/val/test; target metrics: >95% class acc, rep MAE <0.5, <10 ms inference on device.
5) **Deployment (future phases)**
   - Convert to TFLite, apply quantization (FP16/INT8), and integrate with Flutter via BLE data pipeline.

## Common Tasks (Code Snippets)
Preprocess a raw file and write filtered output:
```python
from pathlib import Path
from fitness_ai import preprocess_file

result = preprocess_file(Path("data/raw/deep-squat/athlete_01_deep-squat_2025-11-24T17-06-12_20.csv"))
print(result.output_path, result.missing_ratio)
```

Log metadata into `data/metadata.csv`:
```python
from fitness_ai import parse_filename, append_metadata_row, get_metadata_index_path

meta = parse_filename("athlete_01_deep-squat_2025-11-24T17-06-12_20.csv")
row = meta.as_dict(placement="wrist", firmware="v1.0", labeler="annotatorA")
append_metadata_row(get_metadata_index_path(), row)
```

Detect repetitions and slice segments from a filtered DataFrame:
```python
from fitness_ai import detect_repetitions, slice_segments

peaks = detect_repetitions(df["Acceleration_Z"], fs_hz=104.0)  # choose axis or magnitude
segments = slice_segments(df, peaks, fs_hz=104.0)
for seg in segments:
    seg.dataframe.to_csv(f"data/segmented/deep-squat/{seg.segment_id:02d}.csv", index=False)
```

Use notebooks for quick inspection:
- `notebooks/LowPassFilter.ipynb` — demonstrates filtering a sample file using path helpers.
- `notebooks/SegmentationPeakDetection.ipynb` — runs peak detection/visualization on filtered data.

## Testing
- Prefer `python -m pytest` when tests are added; current utilities are lightweight and can be sanity‑checked in notebooks or small scripts.

## Contributing
- Follow the path and naming helpers (`fitness_ai.utils.paths`) to avoid hard‑coded local paths.
- Keep this README in sync with any changes to preprocessing, segmentation, or data formats.
