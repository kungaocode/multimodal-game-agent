"""ADB executor: screencap + tap/swipe on a real device or emulator.

Usage:
    from executor.adb_executor import AdbExecutor
    adb = AdbExecutor(serial="127.0.0.1:5555")  # or None for auto-detect
    img = adb.screencap()
    adb.tap(640, 360)
    adb.swipe(100, 200, 300, 400, duration_ms=500)

Requires: adb binary in PATH and a connected device/emulator.
Test: adb devices
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdbExecutor:
    """ADB screen capture and input execution for Android devices/emulators."""

    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        screencap_dir: str = "/sdcard/coach_screencap.png",
        timeout: float = 10.0,
    ) -> None:
        self.serial = serial
        self.adb_path = adb_path
        self.screencap_remote = screencap_dir
        self.timeout = timeout
        self._local_tmp = Path(tempfile.gettempdir()) / f"coach_screencap_{uuid.uuid4().hex}.png"

    def _cmd(self, *args: str) -> list[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + list(args)

    def _run(self, *args: str, binary_output: bool = False) -> bytes:
        cmd = self._cmd(*args)
        result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ADB command failed: {' '.join(cmd)} -> {stderr}")
        return result.stdout if binary_output else result.stdout

    def devices(self) -> list[str]:
        """List connected device serials."""
        output = self._run("devices").decode("utf-8")
        lines = output.strip().split("\n")[1:]  # skip header
        serials = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                serials.append(parts[0])
        return serials

    def is_connected(self) -> bool:
        try:
            return len(self.devices()) > 0
        except Exception:
            return False

    def screencap(self) -> Image.Image:
        """Capture a screenshot and return PIL Image."""
        # Method 1: exec-out screencap -p (direct binary, fastest)
        try:
            raw = self._run("exec-out", "screencap", "-p", binary_output=True)
            if len(raw) > 100:
                import io
                return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            logger.debug("exec-out screencap failed: %s, falling back", exc)
        # Method 2: shell screencap to file + pull
        self._run("shell", "screencap", "-p", self.screencap_remote)
        self._run("pull", self.screencap_remote, str(self._local_tmp))
        return Image.open(self._local_tmp).convert("RGB")

    def tap(self, x: int, y: int) -> None:
        """Tap at pixel coordinates."""
        self._run("shell", "input", "tap", str(int(x)), str(int(y)))
        logger.debug("tap(%d, %d)", x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from (x1,y1) to (x2,y2) over duration_ms milliseconds."""
        self._run(
            "shell", "input", "swipe",
            str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms)),
        )
        logger.debug("swipe(%d,%d -> %d,%d, %dms)", x1, y1, x2, y2, duration_ms)

    def deploy_troop(self, troop_bar_x: int, troop_bar_y: int, target_x: int, target_y: int) -> None:
        """Deploy a troop: tap troop icon in bar, then tap deployment location.

        This is a two-step action: select troop type, then place it.
        """
        self.tap(troop_bar_x, troop_bar_y)
        time.sleep(0.3)
        self.tap(target_x, target_y)
        logger.debug("deploy_troop(bar=%d,%d -> target=%d,%d)", troop_bar_x, troop_bar_y, target_x, target_y)

    def back(self) -> None:
        """Press Android back button."""
        self._run("shell", "input", "keyevent", "4")

    def wake(self) -> None:
        """Wake up the screen."""
        self._run("shell", "input", "keyevent", "224")

    def screenshot_array(self) -> np.ndarray:
        """Capture screenshot as numpy RGB array."""
        img = self.screencap()
        return np.array(img)
