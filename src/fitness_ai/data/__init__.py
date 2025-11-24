"""
数据处理模块：提供 Phase 1 所需的采集规范常量、预处理、元数据和分段工具。
"""

from fitness_ai.data.constants import (
    ACCEL_COLS,
    ACCEL_UNIT,
    DATA_COLUMNS,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    DEFAULT_MIN_PROMINENCE_MULTIPLIER,
    DEFAULT_MIN_REP_INTERVAL_S,
    DEFAULT_SEGMENT_POST_S,
    DEFAULT_SEGMENT_PRE_S,
    DEFAULT_WINDOW_S,
    FS_HZ,
    GYRO_COLS,
    GYRO_UNIT,
    TIMESTAMP_COL,
    WINDOW_STEP_S,
)
from fitness_ai.data.metadata import FileMetadata, append_metadata_row, parse_filename
from fitness_ai.data.preprocess import (
    PreprocessConfig,
    PreprocessResult,
    add_time_seconds,
    apply_lowpass_filter,
    interpolate_and_resample,
    load_raw_dataframe,
    preprocess_file,
)
from fitness_ai.data.segmentation import (
    RepSegment,
    SlidingWindow,
    detect_repetitions,
    sliding_windows,
    slice_segments,
)

__all__ = [
    # constants
    "ACCEL_COLS",
    "ACCEL_UNIT",
    "DATA_COLUMNS",
    "DEFAULT_CUTOFF_HZ",
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_MIN_PROMINENCE_MULTIPLIER",
    "DEFAULT_MIN_REP_INTERVAL_S",
    "DEFAULT_SEGMENT_POST_S",
    "DEFAULT_SEGMENT_PRE_S",
    "DEFAULT_WINDOW_S",
    "FS_HZ",
    "GYRO_COLS",
    "GYRO_UNIT",
    "TIMESTAMP_COL",
    "WINDOW_STEP_S",
    # metadata
    "FileMetadata",
    "append_metadata_row",
    "parse_filename",
    # preprocess
    "PreprocessConfig",
    "PreprocessResult",
    "add_time_seconds",
    "apply_lowpass_filter",
    "interpolate_and_resample",
    "load_raw_dataframe",
    "preprocess_file",
    # segmentation
    "RepSegment",
    "SlidingWindow",
    "detect_repetitions",
    "sliding_windows",
    "slice_segments",
]
