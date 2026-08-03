"""Small I/O helpers shared across the pipeline.

atomic_torch_save is the important one: it prevents a corrupted
checkpoint file if the process is killed (Ctrl+C, OOM-kill, power loss)
mid-write. torch.save writes directly to the destination path, so an
interruption during that write leaves a truncated .pt file that will
fail to load on the next run -- exactly when resuming matters most.
Writing to a temp file first and renaming is atomic on both POSIX and
Windows (os.replace), so the destination path only ever points at a
complete file.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_json(obj: dict, path: str | Path) -> None:
    """Write a dict to disk as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def append_csv(path: str | Path, row: dict[str, Any], write_header: bool) -> None:
    """Append a single row to a CSV file, creating parent dirs and header as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Save a torch payload atomically: write to a temp file, then rename.

    Args:
        payload: Dict passed to torch.save.
        path: Final destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, str(path))  # atomic on POSIX and Windows
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
