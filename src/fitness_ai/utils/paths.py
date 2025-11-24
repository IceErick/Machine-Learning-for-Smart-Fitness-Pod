"""Utilities for resolving project-relative filesystem locations."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

DEFAULT_MARKERS: tuple[str, ...] = ("README.md", ".git")


def _has_marker(directory: Path, markers: Iterable[str]) -> bool:
    return any((directory / marker).exists() for marker in markers)


def _walk_upwards(start: Path) -> Iterable[Path]:
    current = start
    while True:
        yield current
        if current.parent == current:
            break
        current = current.parent


def _env_root(markers: Iterable[str]) -> Path | None:
    candidate = os.getenv("FITNESS_AI_ROOT")
    if not candidate:
        return None
    path = Path(candidate).expanduser().resolve()
    return path if _has_marker(path, markers) else None


@lru_cache(maxsize=None)
def get_project_root(markers: Iterable[str] = DEFAULT_MARKERS) -> Path:
    """
    Locate the repository root by searching for marker files/directories.

    Args:
        markers: Filenames or directory names that signal the root.

    Raises:
        RuntimeError: If no directory containing one of the markers is found.
    """

    env_root = _env_root(markers)
    if env_root:
        return env_root

    search_starts = {Path(__file__).resolve(), Path.cwd().resolve()}
    for start in search_starts:
        for candidate in _walk_upwards(start):
            if _has_marker(candidate, markers):
                return candidate

    raise RuntimeError(
        "Unable to locate the project root. "
        "Set FITNESS_AI_ROOT to the repository path if you run from a packaged build."
    )


def get_data_dir() -> Path:
    """Return the repository's data directory."""

    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Return the directory containing raw sensor captures."""

    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """Return the directory containing processed artifacts."""

    data_dir = get_data_dir()
    preferred = data_dir / "processed"
    if preferred.exists():
        return preferred
    legacy = data_dir / "filtered"
    return legacy if legacy.exists() else preferred


def get_segmented_data_dir() -> Path:
    """Return the directory that stores segmented exercise snippets."""

    return get_data_dir() / "segmented"


def get_metadata_index_path() -> Path:
    """Return the default metadata CSV path."""

    return get_data_dir() / "metadata.csv"


def raw_data_path(*relative: str) -> Path:
    """Build a path under ``data/raw``."""

    return get_raw_data_dir().joinpath(*relative)


def processed_data_path(*relative: str) -> Path:
    """Build a path under ``data/processed``."""

    return get_processed_data_dir().joinpath(*relative)


def segmented_data_path(*relative: str) -> Path:
    """Build a path under ``data/segmented``."""

    return get_segmented_data_dir().joinpath(*relative)
