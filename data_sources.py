"""
data_sources.py
Pluggable data-acquisition layer.

The app is built against this common interface so the UI/analysis code
never changes when you swap the acquisition method — manual entry,
simulation, file upload, or (later) a live GPIB poll against the Precio XL.

To add real GPIB support:
    1. pip install pyvisa
    2. Fill in GPIBDataSource._connect() / _poll() with the actual
       Precio XL command strings (see TODOs below).
    3. Select "GPIB (Precio XL) — needs setup" in the source dropdown.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, QTimer, Signal


# ============================================================
# File loading
# ============================================================
def load_profile(file_path):
    """
    Load an Excel / CSV / TXT file and return (x_data, y_data) as numeric
    pandas Series.

    X column (elapsed time) is the first match of:
        "Interval (Min)", "Step", otherwise the first column.
    Y column (height diff) is the first match of:
        "Z Height Difference", "Z Diff", "Z Height", otherwise the second column.
    """
    ext = Path(file_path).suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".txt":
        df = pd.read_csv(file_path, sep=None, engine="python", header=None)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df.columns = [str(c).strip() for c in df.columns]

    x_data = None
    for name in ("Interval (Min)", "Step"):
        if name in df.columns:
            x_data = df[name]
            break
    if x_data is None:
        x_data = df.iloc[:, 0]

    y_data = None
    for name in ("Z Height Difference", "Z Diff", "Z Height"):
        if name in df.columns:
            y_data = df[name]
            break
    if y_data is None:
        if df.shape[1] < 2:
            raise ValueError("File needs at least two columns (time, height diff).")
        y_data = df.iloc[:, 1]

    # Convert to numbers; drop header rows / blanks / text
    pair = pd.DataFrame({
        "x": pd.to_numeric(x_data, errors="coerce"),
        "y": pd.to_numeric(y_data, errors="coerce"),
    }).dropna()

    if pair.empty:
        raise ValueError("No numeric data found in the file.")

    return pair["x"].reset_index(drop=True), pair["y"].reset_index(drop=True)


# ============================================================
# Base class
# ============================================================
class DataSource(QObject):
    """Base class. Emits `reading(elapsed_min, height_diff_um)` whenever a
    new data point is available."""

    reading = Signal(float, float)   # elapsed_min, height_diff_um
    status = Signal(str)             # human-readable status text
    error = Signal(str)
    finished = Signal()              # emitted whenever the source stops

    def __init__(self):
        super().__init__()
        self._running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    @property
    def is_running(self):
        return self._running


# ============================================================
# Manual entry
# ============================================================
class ManualEntryDataSource(DataSource):
    """Operator types in a height-diff reading each interval; this source
    just timestamps and forwards it. Use `submit_reading()` from the UI."""

    def __init__(self):
        super().__init__()
        self._start_time = None

    def start(self):
        self._start_time = time.monotonic()
        self._running = True
        self.status.emit("Manual entry mode — enter each reading as you measure it.")

    def stop(self):
        self._running = False
        self.status.emit("Stopped.")
        self.finished.emit()

    def submit_reading(self, height_diff_um: float):
        if not self._running:
            self.error.emit("Data source not started.")
            return
        elapsed_min = (time.monotonic() - self._start_time) / 60.0
        self.reading.emit(elapsed_min, height_diff_um)


# ============================================================
# Simulated
# ============================================================
class SimulatedDataSource(DataSource):
    """Generates a synthetic exponential-decay soak curve for UI testing,
    on a fast timer (not real 5-min intervals) so you can exercise the
    whole app without hardware."""

    def __init__(self, interval_ms: int = 800, true_A=60.0, true_tau=12.0,
                 true_C=3.0, noise=1.5):
        super().__init__()
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._interval_ms = interval_ms
        self._elapsed_min = 0.0
        self._step_min = 5.0  # mimic real 5-min sampling interval
        self.true_A, self.true_tau, self.true_C, self.noise = true_A, true_tau, true_C, noise

    def start(self):
        self._elapsed_min = 0.0
        self._running = True
        self._timer.start(self._interval_ms)
        self.status.emit("Simulated data source running (accelerated timing).")

    def stop(self):
        self._timer.stop()
        self._running = False
        self.finished.emit()

    def _tick(self):
        y = self.true_A * math.exp(-self._elapsed_min / self.true_tau) + self.true_C
        y += random.gauss(0, self.noise)
        y = max(y, 0.0)
        self.reading.emit(self._elapsed_min, y)
        self._elapsed_min += self._step_min
        if self._elapsed_min > 90:
            self.stop()
            self.status.emit("Simulated run complete.")


# ============================================================
# File upload
# ============================================================
class FileDataSource(DataSource):
    """Loads a whole profile from an Excel / CSV / TXT file. The UI sets
    `file_path` (via a file dialog) before calling start()."""

    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

    def start(self):
        if not self.file_path:
            self.error.emit("No file selected.")
            return

        self._running = True
        try:
            x_data, y_data = load_profile(self.file_path)
            for x, y in zip(x_data, y_data):
                self.reading.emit(float(x), float(y))
            self.status.emit(f"Loaded {len(x_data)} points from {Path(self.file_path).name}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Could not load file: {exc}")

        self.stop()

    def stop(self):
        self._running = False
        self.finished.emit()


# ============================================================
# GPIB (stub)
# ============================================================
class GPIBDataSource(DataSource):
    """
    STUB — wire this up once you have the Precio XL GPIB/SECS command
    reference from TEL, or have reverse-engineered it from existing
    workstation scripts.

    Suggested approach (PyVISA):

        import pyvisa
        rm = pyvisa.ResourceManager()
        self._inst = rm.open_resource("GPIB0::<addr>::INSTR")
        self._inst.timeout = 5000
        self._inst.read_termination = "\\r\\n"
        self._inst.write_termination = "\\r\\n"

        # TODO: replace with actual PrecioXL query command
        response = self._inst.query("<NEEDLE_HEIGHT_QUERY_CMD>")
        height_diff_um = self._parse_height_response(response)

    Poll this on a QTimer at your real interval (default 5 min / 300000 ms).
    """

    def __init__(self, gpib_address: str = "GPIB0::1::INSTR", interval_ms: int = 300_000):
        super().__init__()
        self.gpib_address = gpib_address
        self._interval_ms = interval_ms
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._inst = None
        self._elapsed_min = 0.0

    def start(self):
        try:
            self._connect()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(
                f"GPIB connection failed: {exc}. "
                f"This data source needs the Precio XL command set wired in "
                f"(see data_sources.GPIBDataSource docstring)."
            )
            return
        self._elapsed_min = 0.0
        self._running = True
        self._timer.start(self._interval_ms)
        self.status.emit(f"Polling {self.gpib_address} every {self._interval_ms / 60000:.1f} min.")

    def stop(self):
        self._timer.stop()
        self._running = False
        if self._inst is not None:
            try:
                self._inst.close()
            except Exception:  # noqa: BLE001
                pass
            self._inst = None
        self.status.emit("Stopped.")
        self.finished.emit()

    def _connect(self):
        import pyvisa  # local import so the app runs without pyvisa installed
        rm = pyvisa.ResourceManager()
        self._inst = rm.open_resource(self.gpib_address)
        self._inst.timeout = 5000
        # TODO: set read/write terminators + any init commands per Precio XL manual

    def _poll(self):
        try:
            # TODO: replace with actual query command + response parsing
            raise NotImplementedError(
                "GPIB query not yet implemented — fill in GPIBDataSource._poll() "
                "with the Precio XL command syntax once available."
            )
        except NotImplementedError as exc:
            self.error.emit(str(exc))
            self.stop()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"GPIB read error: {exc}")


SOURCE_REGISTRY = {
    "Manual entry": ManualEntryDataSource,
    "Simulated (for testing)": SimulatedDataSource,
    "File Upload": FileDataSource,
    "GPIB (Precio XL) — needs setup": GPIBDataSource,
}
