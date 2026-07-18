"""
Keyboard input simulation for steering.

Uses Windows SendInput on Windows for reliable key delivery to the focused app.
Falls back to PyAutoGUI on other platforms.
"""

import sys
from typing import Optional

import pyautogui

from steering import SteeringDirection


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUT)]

    _SendInput = ctypes.windll.user32.SendInput
    _SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _SendInput.restype = wintypes.UINT

    VK_LEFT = 0x25
    VK_RIGHT = 0x27
    KEYEVENTF_KEYUP = 0x0002
    INPUT_KEYBOARD = 1


class KeyboardController:
    """
    Press and hold arrow keys based on steering direction.

    Keys are sent to whichever window is focused in Windows (e.g. Notepad or a game).
    Keep the target app focused; clicking the camera window sends keys there instead.
    """

    LEFT_KEY = "left"
    RIGHT_KEY = "right"

    def __init__(self, fail_safe: bool = True, use_sendinput: bool = True) -> None:
        pyautogui.FAILSAFE = fail_safe
        pyautogui.PAUSE = 0
        self._use_sendinput = use_sendinput and sys.platform == "win32"
        self._active_key: Optional[str] = None
        self._last_logged_key: Optional[str] = None

    @property
    def active_key(self) -> Optional[str]:
        """Return the arrow key currently held, or None when no key is pressed."""
        return self._active_key

    def active_key_label(self) -> str:
        """Human-readable label for HUD output."""
        if self._active_key == self.LEFT_KEY:
            return "LEFT ARROW (held)"
        if self._active_key == self.RIGHT_KEY:
            return "RIGHT ARROW (held)"
        return "none"

    def update(self, direction: SteeringDirection, debug: bool = False) -> None:
        """
        Synchronize held keys with the current steering direction.

        - LEFT: hold Left Arrow
        - RIGHT: hold Right Arrow
        - STRAIGHT / UNKNOWN: release any held arrow key
        """
        desired_key = self._desired_key(direction)

        if desired_key == self._active_key:
            return

        self.release_all()
        if desired_key is not None:
            self._key_down(desired_key)
            self._active_key = desired_key

        if debug and desired_key != self._last_logged_key:
            label = desired_key.upper() if desired_key else "RELEASED"
            print(f"[keyboard] {label}")
            self._last_logged_key = desired_key

    def _desired_key(self, direction: SteeringDirection) -> Optional[str]:
        """Map a steering direction to the arrow key that should be held."""
        if direction == SteeringDirection.LEFT:
            return self.LEFT_KEY
        if direction == SteeringDirection.RIGHT:
            return self.RIGHT_KEY
        return None

    def _key_down(self, key: str) -> None:
        """Send a key-down event using the best available backend."""
        if self._use_sendinput:
            self._sendinput_key(key, key_up=False)
        else:
            pyautogui.keyDown(key)

    def _key_up(self, key: str) -> None:
        """Send a key-up event using the best available backend."""
        if self._use_sendinput:
            self._sendinput_key(key, key_up=True)
        else:
            pyautogui.keyUp(key)

    def _sendinput_key(self, key: str, key_up: bool) -> None:
        """Low-level Windows keyboard event for Left/Right arrow keys."""
        vk_code = VK_LEFT if key == self.LEFT_KEY else VK_RIGHT
        flags = KEYEVENTF_KEYUP if key_up else 0
        input_event = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0),
        )
        sent = _SendInput(1, ctypes.byref(input_event), ctypes.sizeof(INPUT))
        if sent != 1:
            pyautogui.keyUp(key) if key_up else pyautogui.keyDown(key)

    def release_all(self) -> None:
        """Release whichever arrow key is currently held, if any."""
        if self._active_key is not None:
            self._key_up(self._active_key)
            self._active_key = None

    def shutdown(self) -> None:
        """Ensure no keys remain pressed when the application exits."""
        self.release_all()
