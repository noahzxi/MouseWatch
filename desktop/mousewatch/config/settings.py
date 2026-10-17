"""Runtime options for the background collector."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def default_output_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        root = str(Path.home() / "AppData" / "Local")
    return Path(root) / "MouseWatch" / "traces.csv"


@dataclass(frozen=True)
class Settings:
    user_id: int
    output_path: Path
    pause_ms: int = 100
    min_distance_px: int = 4
    segment_ms: int = 130
    flush_every: int = 50
    flush_seconds: float = 2.0
    enable_tray: bool = True
    profile_path: Path | None = None


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        prog="mousewatch",
        description=(
            "Collect background mouse movements and clicks in the dataset.csv format."
        ),
        epilog=(
            "Create the master profile with: python -m mousewatch.finetune --user-id ID --trace TRACE. "
            "The running app loads master.joblib beside the trace and shows whether the current user matches it."
        ),
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="Integer written to the idman column (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path. Defaults to %%LOCALAPPDATA%%\\MouseWatch\\traces.csv.",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=100,
        help="Stillness that starts a new movement, in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--segment-ms",
        type=int,
        default=130,
        help="Length of one continuous-stroke row, in milliseconds (default: 130).",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=4,
        help="Drop strokes shorter than this many pixels (default: 4).",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Master profile to compare against. Defaults to master.joblib beside the trace.",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Do not show a tray icon. Stop the process with Ctrl+C.",
    )
    args = parser.parse_args(argv)
    if args.user_id < 0:
        parser.error("--user-id must be zero or greater")
    if args.pause_ms <= 0:
        parser.error("--pause-ms must be positive")
    if args.min_distance < 1:
        parser.error("--min-distance must be positive")
    if args.segment_ms <= 0:
        parser.error("--segment-ms must be positive")
    return Settings(
        user_id=args.user_id,
        output_path=args.output if args.output is not None else default_output_path(),
        pause_ms=args.pause_ms,
        min_distance_px=args.min_distance,
        segment_ms=args.segment_ms,
        enable_tray=not args.no_tray,
        profile_path=args.profile,
    )
