"""Data-parallel helpers: one worker process per GPU, resumable JSONL shards."""

from __future__ import annotations

import json
import os
import subprocess
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator


def batched(items: Iterable, size: int) -> Iterator[list]:
    """Split an iterable into lists of at most `size` items.

    Args:
        items (Iterable): Items.
        size (int): Batch size.

    Returns:
        Iterator[list]: Batches.
    """
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def launch_workers(cmd: list[str], num_workers: int) -> None:
    """Run `cmd` once per GPU with `--worker-index i --num-workers n`, each process seeing only GPU i.

    Args:
        cmd (list[str]): Command and arguments.
        num_workers (int): Number of workers.
    """
    procs = []
    for i in range(num_workers):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        procs.append(subprocess.Popen([*cmd, "--worker-index", str(i), "--num-workers", str(num_workers)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"Worker exit codes: {codes}")


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read records from JSONL files. Truncated lines are skipped.

    Args:
        paths (Iterable[Path]): Files; missing ones are skipped.

    Returns:
        list[dict[str, Any]]: Records, in file order.
    """
    records = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records to a JSONL file, repairing a truncated last line first.

    Args:
        path (Path): File.
        records (Iterable[dict[str, Any]]): Records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
