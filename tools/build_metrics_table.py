#!/usr/bin/env python3
"""Extract val_aupr, val_auroc, val_fpr95 from anomaly result files and print a markdown table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

TARGET_FILES = [
    "all",
    "all_without_potholes",
    "pothole",
    "tiny",
    "small",
    "medium",
    "large",
]

TARGET_METRICS = ["val_aupr", "val_auroc", "val_fpr95"]

FLOAT_PATTERN = r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)"


def extract_metrics(file_path: Path) -> Dict[str, str]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, str] = {}

    for metric in TARGET_METRICS:
        pattern = rf"\b{re.escape(metric)}\s+{FLOAT_PATTERN}"
        match = re.search(pattern, text)
        out[metric] = match.group(1) if match else "N/A"

    return out


def format_markdown_table(rows: List[Dict[str, str]]) -> str:
    header = "| File | val_aupr | val_auroc | val_fpr95 |"
    separator = "|---|---:|---:|---:|"
    lines = [header, separator]

    for row in rows:
        lines.append(
            f"| `{row['file']}` | {row['val_aupr']} | {row['val_auroc']} | {row['val_fpr95']} |"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a markdown table from anomaly metric files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("saved/Train_Carla_New"),
        help="Directory containing files: all, all_without_potholes, tiny, small, medium, large.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.dir

    rows: List[Dict[str, str]] = []
    for name in TARGET_FILES:
        file_path = base_dir / name
        if not file_path.exists():
            rows.append({"file": name, "val_aupr": "N/A", "val_auroc": "N/A", "val_fpr95": "N/A"})
            continue

        metrics = extract_metrics(file_path)
        rows.append({"file": name, **metrics})

    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()

