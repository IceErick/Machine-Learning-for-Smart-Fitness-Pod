"""原始 IMU 数据的加载、补点、滤波与导出工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from fitness_ai.data.constants import (
    ACCEL_COLS,
    DATA_COLUMNS,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    FS_HZ,
    GYRO_COLS,
    TIMESTAMP_COL,
)
from fitness_ai.data.metadata import FileMetadata, parse_filename
from fitness_ai.utils.paths import get_data_dir, processed_data_path


@dataclass(frozen=True)
class PreprocessConfig:
    """预处理参数配置。"""

    fs_hz: float = FS_HZ
    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    filter_order: int = DEFAULT_FILTER_ORDER
    timestamp_col: str = TIMESTAMP_COL
    data_columns: Tuple[str, ...] = DATA_COLUMNS
    min_gap_seconds: float = 2.0 / FS_HZ  # 超过两个周期视为需要补点


@dataclass(frozen=True)
class PreprocessResult:
    """预处理结果与质量指标。"""

    dataframe: pd.DataFrame
    output_path: Path
    expected_samples: int
    interpolated_samples: int
    missing_ratio: float


def load_raw_dataframe(path: Path, expected_columns: Sequence[str] = DATA_COLUMNS) -> pd.DataFrame:
    """读取原始 CSV 并校验字段。"""

    df = pd.read_csv(path)
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"文件缺少列 {missing}; 期望列为 {list(expected_columns)}")
    return df


def add_time_seconds(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL) -> pd.DataFrame:
    """基于时间戳添加 time_s（相对起始秒）。"""

    result = df.copy()
    ts = pd.to_datetime(result[timestamp_col])
    result["time_s"] = (ts - ts.iloc[0]).dt.total_seconds()
    return result


def _period_ns(fs_hz: float) -> int:
    return int(round(1e9 / fs_hz))


def interpolate_and_resample(df: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, dict]:
    """
    将时间轴对齐为固定采样率并线性插值缺口。

    返回补点后的 DataFrame 以及缺失率指标。
    """

    timestamp_col = config.timestamp_col
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    period_ns = _period_ns(config.fs_hz)
    full_index = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq=pd.to_timedelta(period_ns, unit="ns"),
    )

    df_full = df.reindex(full_index)
    interpolated_samples = int(df_full.isna().any(axis=1).sum())
    expected_samples = len(df_full.index)
    missing_ratio = interpolated_samples / expected_samples if expected_samples else 0.0

    df_full = df_full.interpolate(method="time", limit_direction="both")
    df_full.index.name = timestamp_col

    df_full.reset_index(inplace=True)
    df_full = add_time_seconds(df_full, timestamp_col=timestamp_col)

    metrics = {
        "interpolated_samples": interpolated_samples,
        "expected_samples": expected_samples,
        "missing_ratio": missing_ratio,
    }
    return df_full, metrics


def apply_lowpass_filter(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    fs_hz: float = FS_HZ,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> pd.DataFrame:
    """对指定列进行零相位巴特沃斯低通滤波。"""

    result = df.copy()
    cols = list(columns) if columns is not None else list(ACCEL_COLS + GYRO_COLS)
    nyquist = 0.5 * fs_hz
    norm_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(filter_order, norm_cutoff, btype="low", analog=False)
    for col in cols:
        if col not in result.columns:
            continue
        result[col] = signal.filtfilt(b, a, result[col].to_numpy())
    return result


def _default_filtered_path(raw_path: Path, metadata: Optional[FileMetadata]) -> Path:
    if metadata:
        return processed_data_path(metadata.exercise, raw_path.name)
    # 回退：沿用 raw 下的相对目录
    data_root = get_data_dir()
    raw_dir = Path(raw_path).parent
    try:
        relative = raw_dir.relative_to(data_root / "raw")
    except ValueError:
        relative = raw_dir.relative_to(data_root)
    return processed_data_path(relative, raw_path.name)


def preprocess_file(
    raw_path: Path,
    output_path: Path | None = None,
    config: PreprocessConfig = PreprocessConfig(),
    write_output: bool = True,
) -> PreprocessResult:
    """
    从原始 CSV 到滤波数据的完整预处理流程。

    - 固定采样率插值
    - time_s 生成
    - 低通滤波
    - 写出到 data/filtered 下对应目录
    """

    metadata: Optional[FileMetadata]
    try:
        metadata = parse_filename(raw_path)
    except ValueError:
        metadata = None

    df_raw = load_raw_dataframe(raw_path, expected_columns=config.data_columns)
    df_resampled, metrics = interpolate_and_resample(df_raw, config=config)
    df_filtered = apply_lowpass_filter(
        df_resampled,
        columns=ACCEL_COLS + GYRO_COLS,
        fs_hz=config.fs_hz,
        cutoff_hz=config.cutoff_hz,
        filter_order=config.filter_order,
    )

    dest = output_path or _default_filtered_path(raw_path, metadata)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if write_output:
        df_filtered.to_csv(dest, index=False)

    return PreprocessResult(
        dataframe=df_filtered,
        output_path=dest,
        expected_samples=metrics["expected_samples"],
        interpolated_samples=metrics["interpolated_samples"],
        missing_ratio=metrics["missing_ratio"],
    )
