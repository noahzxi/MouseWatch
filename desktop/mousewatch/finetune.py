"""Fine-tune the trained PointerPrint model on a desktop trace.

Run from the repository after the collector has written traces.csv:

    python -m mousewatch.finetune --user-id 1

The population model stays in place. This command adds the desktop user, or
moves an existing user toward the strokes in their trace, and writes a
personal model beside the trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mousewatch.config.settings import default_output_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_pointerprint():
    source = repo_root() / "src"
    if source.is_dir():
        entry = str(source)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from pointerprint.dataset import load_actions
    from pointerprint.features import feature_matrix
    from pointerprint.model import PointerPrintModel
    from pointerprint.personal import PersonalTemplate

    return load_actions, feature_matrix, PointerPrintModel, PersonalTemplate


def default_base_model() -> Path:
    return repo_root() / "models" / "pointerprint.joblib"


def fine_tune_trace(
    trace_path: str | Path,
    user_id: int | str,
    base_model: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, float | int | bool | str]:
    """Load `base_model`, update `user_id` from the trace, and save the result."""
    load_actions, feature_matrix, PointerPrintModel, PersonalTemplate = _import_pointerprint()
    trace = Path(trace_path)
    base = Path(base_model)
    if not trace.is_file():
        raise FileNotFoundError(f"trace not found: {trace}")
    if not base.is_file():
        raise FileNotFoundError(f"base model not found: {base}")
    label = str(user_id)
    groups = load_actions(trace, progress_every=0)
    if label not in groups:
        known = ", ".join(sorted(groups)) or "(none)"
        raise ValueError(f"no rows for user {label} in {trace}; found {known}")
    model = PointerPrintModel.load(base)
    windows = feature_matrix(
        groups[label],
        window=model.window,
        stride=model.stride,
        session_gap_seconds=model.session_gap_seconds,
    )
    summary = model.fine_tune(label, windows)
    model.owner_id = label
    personal_actions = groups[label]
    motion = trace.with_name(trace.stem + ".motion.csv")
    if motion.is_file():
        from mousewatch.storage.motion import load_motion_actions

        try:
            rich = load_motion_actions(motion, label)
        except ValueError:
            rich = None
        if rich is not None and rich.shape[0] >= 3:
            personal_actions = rich
    model.personal_template = PersonalTemplate.fit(personal_actions)
    destination = Path(output_path) if output_path is not None else trace.with_name("master.joblib")
    model.save(destination)
    summary["model_path"] = str(destination)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mousewatch.finetune",
        description="Fine-tune the trained mouse model on a user's collected trace.",
    )
    parser.add_argument("--user-id", required=True, help="idman value in the trace.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Collector CSV. Defaults to %%LOCALAPPDATA%%\\MouseWatch\\traces.csv.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Trained PointerPrint model. Defaults to models/pointerprint.joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the master profile. Defaults to master.joblib next to the trace.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = args.trace if args.trace is not None else default_output_path()
    base = args.base_model if args.base_model is not None else default_base_model()
    try:
        summary = fine_tune_trace(trace, args.user_id, base, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"Fine-tuned user {summary['user_id']} on {summary['windows']} windows. "
        f"Wrote {summary['model_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
