"""Background collector entry point.

Run from the desktop directory:

    pythonw -m mousewatch

Or double-click MouseWatch.bat. Rows are appended to
%LOCALAPPDATA%\\MouseWatch\\traces.csv unless --output is set.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import time
from ctypes import wintypes
from pathlib import Path

from mousewatch.collector.buffer import MovementAssembler, Record
from mousewatch.collector.listener import MouseListener, enable_dpi_awareness
from mousewatch.config.settings import Settings, parse_args
from mousewatch.display import StatusWindow
from mousewatch.identity import LiveIdentity, missing_profile_status
from mousewatch.storage.traces import TraceWriter

kernel32 = ctypes.windll.kernel32
_ERROR_ALREADY_EXISTS = 183
_mutex_handle = None


def _configure_win32_error_api() -> None:
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetLastError.restype = wintypes.DWORD
    kernel32.SetLastError.argtypes = [wintypes.DWORD]


def acquire_single_instance() -> bool:
    """Return False when another collector process is already running."""
    global _mutex_handle
    _configure_win32_error_api()
    kernel32.SetLastError(0)
    _mutex_handle = kernel32.CreateMutexW(None, False, "Local\\MouseWatchCollector")
    if not _mutex_handle:
        return True
    if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(_mutex_handle)
        _mutex_handle = None
        return False
    return True


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _profile_path(settings: Settings) -> Path:
    if settings.profile_path is not None:
        return settings.profile_path
    return settings.output_path.with_name("master.joblib")


def _open_identity(path: Path) -> LiveIdentity | None:
    if not path.is_file():
        return None
    return LiveIdentity.load(path)


def _publish(display: StatusWindow | None, listener: MouseListener, status) -> None:
    if display is not None:
        display.update(status)
    listener.set_tip(status.headline)


def _apply(assembler: MovementAssembler, event) -> list[Record]:
    kind = event[0]
    if kind == "move":
        _, dx, dy, when_ns = event
        return assembler.add_move(dx, dy, when_ns)
    if kind == "button":
        _, button, pressed, when_ns = event
        return assembler.add_button(button, pressed, when_ns)
    return []


def run(settings: Settings) -> int:
    if sys.platform != "win32":
        print("MouseWatch collects mouse data on Windows only.", file=sys.stderr)
        return 1
    if not acquire_single_instance():
        print("MouseWatch is already running.", file=sys.stderr)
        return 0

    enable_dpi_awareness()
    _configure_logging(settings.output_path.with_name("collector.log"))
    logging.info(
        "collecting user %s to %s", settings.user_id, settings.output_path
    )
    profile_path = _profile_path(settings)
    try:
        identity = _open_identity(profile_path)
    except (OSError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"MouseWatch is collecting mouse data for user {settings.user_id}.")
    print(f"Writing {settings.output_path}")
    if identity is None:
        print(f"No master profile at {profile_path}.")
        print(
            "Create one with: python -m mousewatch.finetune --user-id",
            settings.user_id,
            "--trace",
            settings.output_path,
        )
    else:
        print(f"Comparing movement with master profile {profile_path} (user {identity.owner_id}).")
    if settings.enable_tray:
        print("Right-click the tray icon and choose Exit to stop.")
    else:
        print("Press Ctrl+C to stop.")

    events: queue.SimpleQueue = queue.SimpleQueue()
    listener = MouseListener(events, enable_tray=settings.enable_tray)
    display = StatusWindow(on_close=listener.request_stop)
    if not display.start():
        display = None
        print("The status window could not be opened. The tray icon still shows the verdict.")
    assembler = MovementAssembler(
        user_id=settings.user_id,
        pause_ns=settings.pause_ms * 1_000_000,
        min_distance_px=settings.min_distance_px,
        segment_ns=settings.segment_ms * 1_000_000,
    )
    code = 0
    writer: TraceWriter | None = None
    try:
        listener.start()
        opening = (
            identity.status()
            if identity is not None
            else missing_profile_status(profile_path)
        )
        _publish(display, listener, opening)
        writer = TraceWriter(settings.output_path, settings.flush_every)

        def publish_records(records: list[Record]) -> None:
            assert writer is not None
            for record in records:
                writer.write(record)
                if identity is not None:
                    _publish(display, listener, identity.add(record))

        while True:
            timeout = (
                settings.pause_ms / 1000
                if assembler.has_open_movement
                else settings.flush_seconds
            )
            try:
                event = events.get(timeout=timeout)
            except queue.Empty:
                publish_records(assembler.flush(time.time_ns()))
                writer.flush()
                continue
            if event is None:
                break
            publish_records(_apply(assembler, event))
    except KeyboardInterrupt:
        logging.info("interrupted")
    except Exception:
        logging.exception("collector failed")
        code = 1
    finally:
        listener.request_stop()
        listener.join(timeout=2)
        if display is not None:
            display.close()
        if writer is not None:
            try:
                while True:
                    event = events.get_nowait()
                    if event is None:
                        continue
                    for record in _apply(assembler, event):
                        writer.write(record)
            except queue.Empty:
                pass
            for record in assembler.flush(time.time_ns(), force=True):
                writer.write(record)
            rows = writer.rows_written
            writer.close()
            logging.info("stopped after %s rows", rows)
    return code


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
