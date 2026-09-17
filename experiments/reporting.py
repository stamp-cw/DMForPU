"""Shared human-readable reporting helpers for experiment entry points."""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def environment_metadata(device: str = "CPU") -> dict[str, object]:
    metadata: dict[str, object] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "device": device,
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["cuda"] = torch.version.cuda
            metadata["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        metadata["torch"] = "not installed"
    return metadata


def _display(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown_report(
    path: Path,
    title: str,
    environment: Mapping[str, object],
    protocol: Mapping[str, object],
    tables: Iterable[tuple[str, Sequence[str], Sequence[Sequence[object]]]],
    notes: Sequence[str] = (),
) -> None:
    lines = [f"# {title}", "", "## 运行环境", ""]
    lines.extend(f"- {key}: `{_display(value)}`" for key, value in environment.items())
    lines.extend(["", "## 实验协议", ""])
    lines.extend(f"- {key}: `{_display(value)}`" for key, value in protocol.items())
    for heading, headers, rows in tables:
        lines.extend(["", f"## {heading}", "", "| " + " | ".join(headers) + " |"])
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(_display(value) for value in row) + " |" for row in rows)
    if notes:
        lines.extend(["", "## 说明", ""])
        lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
