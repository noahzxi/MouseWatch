"""System-tray icon so the background collector can be closed without a window."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
shell32 = ctypes.windll.shell32

NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIF_MESSAGE = 0x1
NIF_ICON = 0x2
NIF_TIP = 0x4
WM_COMMAND = 0x0111
WM_RBUTTONUP = 0x0205
WM_CONTEXTMENU = 0x007B
WM_NULL = 0x0000
TPM_RIGHTBUTTON = 0x0002
MF_STRING = 0x0000
IDI_APPLICATION = 32512

CALLBACK_MESSAGE = 0x8000 + 20
EXIT_COMMAND = 100


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
    ]


user32.LoadIconW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
user32.LoadIconW.restype = wintypes.HICON
user32.CreatePopupMenu.argtypes = []
user32.CreatePopupMenu.restype = wintypes.HMENU
user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_uint, wintypes.LPCWSTR]
user32.AppendMenuW.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.TrackPopupMenu.argtypes = [
    wintypes.HMENU,
    wintypes.UINT,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    ctypes.c_void_p,
]
user32.TrackPopupMenu.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.DestroyMenu.argtypes = [wintypes.HMENU]
user32.DestroyMenu.restype = wintypes.BOOL
shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
shell32.Shell_NotifyIconW.restype = wintypes.BOOL


class TrayIcon:
    def __init__(self, on_exit) -> None:
        self._on_exit = on_exit
        self._hwnd: wintypes.HWND | None = None
        self._added = False
        self._tip = "MouseWatch is collecting mouse data"
        self._icon = user32.LoadIconW(None, IDI_APPLICATION)

    def add(self, hwnd) -> bool:
        self._hwnd = hwnd
        nid = self._data(hwnd)
        self._added = bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid)))
        return self._added

    def set_tip(self, text: str) -> None:
        self._tip = text[:127]
        if not self._added or self._hwnd is None:
            return
        nid = self._data(self._hwnd)
        shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

    def remove(self) -> None:
        if not self._added or self._hwnd is None:
            return
        nid = self._data(self._hwnd)
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
        self._added = False

    def handle(self, hwnd, msg: int, wparam: int, lparam: int) -> bool:
        if msg == CALLBACK_MESSAGE and lparam in (WM_RBUTTONUP, WM_CONTEXTMENU):
            self._popup(hwnd)
            return True
        if msg == WM_COMMAND and (wparam & 0xFFFF) == EXIT_COMMAND:
            self._on_exit()
            return True
        return False

    def _popup(self, hwnd) -> None:
        menu = user32.CreatePopupMenu()
        if not menu:
            return
        try:
            user32.AppendMenuW(menu, MF_STRING, EXIT_COMMAND, "Exit")
            point = POINT()
            user32.GetCursorPos(ctypes.byref(point))
            user32.SetForegroundWindow(hwnd)
            user32.TrackPopupMenu(
                menu, TPM_RIGHTBUTTON, point.x, point.y, 0, hwnd, None
            )
            user32.PostMessageW(hwnd, WM_NULL, 0, 0)
        finally:
            user32.DestroyMenu(menu)

    def _data(self, hwnd) -> NOTIFYICONDATAW:
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = CALLBACK_MESSAGE
        nid.hIcon = self._icon
        nid.szTip = self._tip
        return nid
