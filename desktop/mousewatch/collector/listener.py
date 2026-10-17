"""Background mouse capture using Raw Input.

Raw Input is posted to a hidden window. Unlike a low-level mouse hook, a slow
collector does not stall the system pointer. Cursor deltas come from
GetCursorPos so distance is in screen pixels, after pointer speed and acceleration.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes

from mousewatch.tray.status import POINT, TrayIcon

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_INPUT = 0x00FF
WM_QUIT = 0x0012
_RESAMPLE_CURSOR = 0x8000 + 21
MOUSE_MOVE_ABSOLUTE = 0x01
PM_NOREMOVE = 0x0000
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
RIDEV_REMOVE = 0x00000001
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02
WS_POPUP = 0x80000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
ERROR_CLASS_ALREADY_EXISTS = 1410

# Button flags in RAWMOUSE.usButtonFlags. The wheel is not a click.
_BUTTONS = (
    (0x0001, 0x0002, "left"),
    (0x0004, 0x0008, "right"),
    (0x0010, 0x0020, "middle"),
)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RawMouseButtons(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class _RawMouseUnion(ctypes.Union):
    _anonymous_ = ("buttons",)
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("buttons", _RawMouseButtons),
    ]


class RAWMOUSE(ctypes.Structure):
    """MSVC layout: the button union is aligned, so it begins two bytes after usFlags."""

    _anonymous_ = ("buttons",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("buttons", _RawMouseUnion),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


def _bind() -> None:
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = LRESULT
    user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
    user32.RegisterClassExW.restype = wintypes.ATOM
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.UnregisterClassW.restype = wintypes.BOOL
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.GetMessageW.restype = ctypes.c_int
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.PostThreadMessageW.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE),
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.RegisterRawInputDevices.restype = wintypes.BOOL
    user32.GetRawInputData.argtypes = [
        ctypes.c_void_p,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.UINT),
        wintypes.UINT,
    ]
    user32.GetRawInputData.restype = wintypes.UINT
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    kernel32.GetLastError.argtypes = []
    kernel32.GetLastError.restype = wintypes.DWORD
    kernel32.SetLastError.argtypes = [wintypes.DWORD]
    kernel32.SetLastError.restype = None


_bind()


def button_changes(flags: int) -> list[tuple[str, bool]]:
    """Translate RAWMOUSE button flags into (button, pressed) pairs."""
    changes: list[tuple[str, bool]] = []
    for down, up, name in _BUTTONS:
        if flags & down:
            changes.append((name, True))
        if flags & up:
            changes.append((name, False))
    return changes


def enable_dpi_awareness() -> None:
    """Ask for physical pixels so stroke length matches the on-screen cursor."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except OSError:
        return


class MouseListener:
    """Queue ('move', dx, dy, when_ns) and ('button', name, pressed, when_ns)."""

    def __init__(self, events, enable_tray: bool = True) -> None:
        self._events = events
        self._enable_tray = enable_tray
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._hwnd = None
        self._instance = None
        self._class_name = "MouseWatchRawInput"
        self._proc = WNDPROC(self._wndproc)
        self._tray: TrayIcon | None = None
        self._last_cursor: tuple[int, int] | None = None
        self._resample_left = 0
        self._buffer = RAWINPUT()

    def start(self) -> None:
        if ctypes.sizeof(ctypes.c_void_p) == 8 and ctypes.sizeof(RAWINPUTHEADER) != 24:
            raise RuntimeError("RAWINPUTHEADER layout does not match 64-bit Windows")
        self._thread = threading.Thread(
            target=self._thread_main,
            name="mousewatch-listener",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise TimeoutError("mouse listener did not start")
        if self._error is not None:
            raise self._error

    def set_tip(self, text: str) -> None:
        if self._tray is not None:
            self._tray.set_tip(text)

    def request_stop(self) -> None:
        self._events.put(None)
        thread_id = self._thread_id
        if thread_id:
            user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _thread_main(self) -> None:
        self._thread_id = int(kernel32.GetCurrentThreadId())
        try:
            self._setup()
            self._ready.set()
            self._loop()
        except Exception as exc:
            self._error = exc
            logging.exception("mouse listener failed")
            self._events.put(None)
            self._ready.set()
        finally:
            self._teardown()

    def _setup(self) -> None:
        enable_dpi_awareness()
        self._instance = kernel32.GetModuleHandleW(None)
        window_class = WNDCLASSEXW()
        window_class.cbSize = ctypes.sizeof(WNDCLASSEXW)
        window_class.lpfnWndProc = self._proc
        window_class.hInstance = self._instance
        window_class.lpszClassName = self._class_name
        kernel32.SetLastError(0)
        atom = user32.RegisterClassExW(ctypes.byref(window_class))
        if not atom and kernel32.GetLastError() != ERROR_CLASS_ALREADY_EXISTS:
            raise ctypes.WinError()

        self._hwnd = user32.CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            self._class_name,
            "MouseWatch",
            WS_POPUP,
            0,
            0,
            0,
            0,
            None,
            None,
            self._instance,
            None,
        )
        if not self._hwnd:
            raise ctypes.WinError()

        device = RAWINPUTDEVICE(
            HID_USAGE_PAGE_GENERIC,
            HID_USAGE_GENERIC_MOUSE,
            RIDEV_INPUTSINK,
            self._hwnd,
        )
        if not user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)
        ):
            raise ctypes.WinError()

        if self._enable_tray:
            self._tray = TrayIcon(self.request_stop)
            if not self._tray.add(self._hwnd):
                logging.error("tray icon was not added; stop the collector from the console")
                self._tray = None

        self._last_cursor = _cursor_position()
        message = MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)

    def _loop(self) -> None:
        message = MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == 0 or result == -1:
                return
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

    def _teardown(self) -> None:
        device = RAWINPUTDEVICE(
            HID_USAGE_PAGE_GENERIC,
            HID_USAGE_GENERIC_MOUSE,
            RIDEV_REMOVE,
            None,
        )
        user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)
        )
        if self._tray is not None:
            self._tray.remove()
            self._tray = None
        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None
        if self._instance is not None:
            user32.UnregisterClassW(self._class_name, self._instance)

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_INPUT:
                self._on_input(lparam)
                return 0
            if msg == _RESAMPLE_CURSOR:
                self._resample_cursor()
                return 0
            if self._tray is not None and self._tray.handle(hwnd, msg, wparam, lparam):
                return 0
        except Exception:
            logging.exception("mouse window procedure failed")
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _on_input(self, handle) -> None:
        size = wintypes.UINT(ctypes.sizeof(self._buffer))
        result = user32.GetRawInputData(
            ctypes.c_void_p(handle),
            RID_INPUT,
            ctypes.byref(self._buffer),
            ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if result in (0, 0xFFFFFFFF):
            return
        if self._buffer.header.dwType != RIM_TYPEMOUSE:
            return
        mouse = self._buffer.mouse
        when_ns = time.time_ns()
        # Raw Input often arrives before the system cursor moves. Sample now, and
        # once more on a later message if this packet carried motion we could not see.
        saw_motion = bool(mouse.usFlags & MOUSE_MOVE_ABSOLUTE) or bool(
            mouse.lLastX or mouse.lLastY
        )
        if saw_motion and not self._emit_cursor(when_ns):
            self._schedule_resample()
        for name, pressed in button_changes(mouse.usButtonFlags):
            self._events.put(("button", name, pressed, when_ns))

    def _emit_cursor(self, when_ns: int) -> bool:
        position = _cursor_position()
        if position is None:
            return False
        previous = self._last_cursor
        self._last_cursor = position
        if previous is None:
            return False
        dx = position[0] - previous[0]
        dy = position[1] - previous[1]
        if dx == 0 and dy == 0:
            return False
        self._resample_left = 0
        self._events.put(("move", dx, dy, when_ns))
        return True

    def _schedule_resample(self) -> None:
        if self._resample_left > 0 or not self._hwnd:
            return
        self._resample_left = 2
        user32.PostMessageW(self._hwnd, _RESAMPLE_CURSOR, 0, 0)

    def _resample_cursor(self) -> None:
        if self._resample_left <= 0:
            return
        self._resample_left -= 1
        if self._emit_cursor(time.time_ns()):
            return
        if self._resample_left > 0 and self._hwnd:
            user32.PostMessageW(self._hwnd, _RESAMPLE_CURSOR, 0, 0)


def _cursor_position() -> tuple[int, int] | None:
    point = POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    return int(point.x), int(point.y)
