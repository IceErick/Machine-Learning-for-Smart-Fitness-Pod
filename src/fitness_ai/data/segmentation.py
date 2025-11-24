"""基于 Phase 1 规划的滑窗与峰值分段工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from fitness_ai.data.constants import (
    DEFAULT_MIN_PROMINENCE_MULTIPLIER,
    DEFAULT_MIN_REP_INTERVAL_S,
    DEFAULT_SEGMENT_POST_S,
    DEFAULT_SEGMENT_PRE_S,
    DEFAULT_WINDOW_S,
    FS_HZ,
    WINDOW_STEP_S,
)


@dataclass(frozen=True)
class SlidingWindow:
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class RepSegment:
    segment_id: int
    peak_index: int
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    dataframe: pd.DataFrame


def sliding_windows(
    df: pd.DataFrame,
    fs_hz: float = FS_HZ,
    window_s: float = DEFAULT_WINDOW_S,
    step_s: float = WINDOW_STEP_S,
    require_ratio: float = 0.6,
    time_col: str = "time_s",
) -> List[SlidingWindow]:
    """
    生成定长滑窗，缺失率高于 require_ratio 的窗口会被跳过。
    """

    if time_col not in df.columns:
        raise ValueError(f"DataFrame 缺少 {time_col} 列")

    window_samples = int(round(window_s * fs_hz))
    step_samples = int(round(step_s * fs_hz))
    windows: List[SlidingWindow] = []

    for start in range(0, len(df) - window_samples + 1, step_samples):
        end = start + window_samples
        slice_df = df.iloc[start:end]
        if slice_df.isna().any(axis=1).mean() > (1 - require_ratio):
            continue
        windows.append(
            SlidingWindow(
                start_index=start,
                end_index=end,
                start_time_s=float(slice_df[time_col].iloc[0]),
                end_time_s=float(slice_df[time_col].iloc[-1]),
            )
        )
    return windows


def detect_repetitions(
    signal_values: Sequence[float],
    fs_hz: float = FS_HZ,
    min_interval_s: float = DEFAULT_MIN_REP_INTERVAL_S,
    prominence: float | None = None,
    prominence_multiplier: float = DEFAULT_MIN_PROMINENCE_MULTIPLIER,
) -> np.ndarray:
    """
    在单轴信号上检测峰/谷，默认基于标准差估计 prominence。
    """

    values = np.asarray(signal_values)
    if prominence is None:
        prominence = np.std(values) * prominence_multiplier
    distance = int(fs_hz * min_interval_s)
    peaks, _ = find_peaks(values, distance=distance, prominence=prominence)
    return peaks


def slice_segments(
    df: pd.DataFrame,
    peaks: Iterable[int],
    fs_hz: float = FS_HZ,
    pre_window_s: float = DEFAULT_SEGMENT_PRE_S,
    post_window_s: float = DEFAULT_SEGMENT_POST_S,
    time_col: str = "time_s",
) -> List[RepSegment]:
    """
    按峰值前后时间窗口切分单次重复片段。
    """

    if time_col not in df.columns:
        raise ValueError(f"DataFrame 缺少 {time_col} 列")

    pre = int(pre_window_s * fs_hz)
    post = int(post_window_s * fs_hz)
    segments: List[RepSegment] = []
    for idx, peak in enumerate(peaks):
        start = max(0, peak - pre)
        end = min(len(df), peak + post)
        seg_df = df.iloc[start:end].reset_index(drop=True)
        segments.append(
            RepSegment(
                segment_id=idx,
                peak_index=int(peak),
                start_index=int(start),
                end_index=int(end),
                start_time_s=float(df[time_col].iloc[start]),
                end_time_s=float(df[time_col].iloc[end - 1]),
                dataframe=seg_df,
            )
        )
    return segments
