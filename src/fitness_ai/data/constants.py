"""Phase 1 采集与预处理相关的常量定义。"""

from __future__ import annotations

from typing import Final, Tuple

FS_HZ: Final[float] = 104.0
ACCEL_UNIT: Final[str] = "m/s^2"
GYRO_UNIT: Final[str] = "rad/s"

TIMESTAMP_COL: Final[str] = "Timestamp"
ACCEL_COLS: Tuple[str, str, str] = ("Acceleration_X", "Acceleration_Y", "Acceleration_Z")
GYRO_COLS: Tuple[str, str, str] = ("Gyro_X", "Gyro_Y", "Gyro_Z")
DATA_COLUMNS: Tuple[str, ...] = (TIMESTAMP_COL,) + ACCEL_COLS + GYRO_COLS

DEFAULT_CUTOFF_HZ: Final[float] = 10.0
DEFAULT_FILTER_ORDER: Final[int] = 4

DEFAULT_WINDOW_S: Final[float] = 2.0
WINDOW_STEP_S: Final[float] = 0.5
DEFAULT_MIN_REP_INTERVAL_S: Final[float] = 1.0
DEFAULT_MIN_PROMINENCE_MULTIPLIER: Final[float] = 3.0
DEFAULT_SEGMENT_PRE_S: Final[float] = 0.8
DEFAULT_SEGMENT_POST_S: Final[float] = 1.0
