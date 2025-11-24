"""Phase 1 元数据与命名规范辅助工具。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from fitness_ai.data.constants import ACCEL_UNIT, FS_HZ, GYRO_UNIT

FILENAME_PATTERN = re.compile(
    r"^athlete_(?P<athlete>[\w-]+)_(?P<exercise>[\w-]+)_(?P<dt>[^_]+)_(?P<reps>\d+)\.csv$"
)

METADATA_FIELDS = [
    "file_name",
    "athlete_id",
    "exercise",
    "reps",
    "datetime_iso",
    "placement",
    "firmware",
    "fs_hz",
    "accel_unit",
    "gyro_unit",
    "orientation_notes",
    "labeler",
    "notes",
]


def _normalize_datetime(dt_str: str) -> str:
    """
    将文件名中的日期字符串转换为 ISO8601 兼容格式。

    例：2025-11-24T17-06-12 -> 2025-11-24T17:06:12
    """

    if "T" not in dt_str:
        return dt_str
    date_part, time_part = dt_str.split("T", 1)
    # 文件名中时间常用连字符分隔，统一替换为冒号
    time_part = time_part.replace("-", ":")
    return f"{date_part}T{time_part}"


def _parse_datetime(dt_str: str) -> datetime:
    normalized = _normalize_datetime(dt_str)
    return datetime.fromisoformat(normalized)


@dataclass(frozen=True)
class FileMetadata:
    """从文件名解析的核心元数据。"""

    athlete_id: str
    exercise: str
    reps: int
    datetime_iso: str
    file_name: str

    @classmethod
    def from_path(cls, path: Path) -> "FileMetadata":
        return parse_filename(path)

    def as_dict(
        self,
        placement: str = "",
        firmware: str = "",
        orientation_notes: str = "",
        labeler: str = "",
        notes: str = "",
        fs_hz: float = FS_HZ,
        accel_unit: str = ACCEL_UNIT,
        gyro_unit: str = GYRO_UNIT,
    ) -> Dict[str, str]:
        """转为写入 metadata CSV 所需的字典。"""

        return {
            "file_name": self.file_name,
            "athlete_id": self.athlete_id,
            "exercise": self.exercise,
            "reps": str(self.reps),
            "datetime_iso": self.datetime_iso,
            "placement": placement,
            "firmware": firmware,
            "fs_hz": str(fs_hz),
            "accel_unit": accel_unit,
            "gyro_unit": gyro_unit,
            "orientation_notes": orientation_notes,
            "labeler": labeler,
            "notes": notes,
        }


def parse_filename(path: str | Path) -> FileMetadata:
    """解析命名规则 athlete_<id>_<exercise>_<datetime>_<reps>.csv。"""

    path = Path(path)
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            f"文件名不符合命名规范：{path.name}，应为 athlete_<编号>_<运动>_<时间日期>_<往返次数>.csv"
        )

    athlete = match.group("athlete")
    exercise = match.group("exercise")
    dt_raw = match.group("dt")
    reps = int(match.group("reps"))
    dt_iso = _normalize_datetime(dt_raw)
    # 校验解析是否成功
    _parse_datetime(dt_raw)

    return FileMetadata(
        athlete_id=athlete,
        exercise=exercise,
        reps=reps,
        datetime_iso=dt_iso,
        file_name=path.name,
    )


def append_metadata_row(metadata_csv: Path, row: Dict[str, str]) -> None:
    """
    追加元数据行到 metadata 索引文件。

    如果文件不存在会自动写入表头。
    """

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metadata_csv.exists()
    with metadata_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
