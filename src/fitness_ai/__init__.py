"""
Core fitness_ai package exports.

Exposes convenience utilities so downstream notebooks and scripts can
import from fitness_ai without worrying about the underlying layout.
"""

from fitness_ai.data import (
    ACCEL_COLS,
    DATA_COLUMNS,
    FS_HZ,
    GYRO_COLS,
    TIMESTAMP_COL,
    FileMetadata,
    PreprocessConfig,
    PreprocessResult,
    add_time_seconds,
    append_metadata_row,
    apply_lowpass_filter,
    detect_repetitions,
    interpolate_and_resample,
    load_raw_dataframe,
    parse_filename,
    preprocess_file,
    slice_segments,
    sliding_windows,
)
from fitness_ai.utils.paths import (
    get_data_dir,
    get_metadata_index_path,
    get_processed_data_dir,
    get_project_root,
    get_raw_data_dir,
    get_segmented_data_dir,
    processed_data_path,
    raw_data_path,
    segmented_data_path,
)

__all__ = [
    # constants
    "ACCEL_COLS",
    "DATA_COLUMNS",
    "FS_HZ",
    "GYRO_COLS",
    "TIMESTAMP_COL",
    # filesystem
    "get_project_root",
    "get_data_dir",
    "get_raw_data_dir",
    "get_processed_data_dir",
    "get_segmented_data_dir",
    "get_metadata_index_path",
    "raw_data_path",
    "processed_data_path",
    "segmented_data_path",
    # metadata
    "FileMetadata",
    "parse_filename",
    "append_metadata_row",
    # preprocess
    "PreprocessConfig",
    "PreprocessResult",
    "load_raw_dataframe",
    "add_time_seconds",
    "interpolate_and_resample",
    "apply_lowpass_filter",
    "preprocess_file",
    # segmentation
    "detect_repetitions",
    "sliding_windows",
    "slice_segments",
]
