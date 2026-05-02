"""
Calibration monitor: async drift detection with configurable callbacks.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class CalibrationMonitor:
    """
    Background thread that periodically checks for stale calibrations.

    When a stale parameter is detected, the registered *alert_fn* is called
    with the qubit label and parameter name.  Optionally, a *recalibrate_fn*
    can be registered to automatically trigger re-calibration.

    Parameters
    ----------
    store : CalibrationStore
        The store to monitor.
    poll_interval_s : float
        How often (in seconds) to check for stale entries (default 300 s = 5 min).
    alert_fn : callable or None
        Called as ``alert_fn(qubit, key)`` when a stale entry is found.
    recalibrate_fn : callable or None
        Called as ``recalibrate_fn(qubit, key)`` to trigger recalibration.
        Only invoked when ``auto_recalibrate=True``.

    Example
    -------
    >>> monitor = CalibrationMonitor(store, poll_interval_s=60,
    ...     alert_fn=lambda q, k: print(f"STALE: {q}/{k}"))
    >>> monitor.start()
    >>> # ... run experiments ...
    >>> monitor.stop()
    """

    def __init__(
        self,
        store,
        poll_interval_s: float = 300.0,
        alert_fn: Callable[[str, str], None] | None = None,
        recalibrate_fn: Callable[[str, str], None] | None = None,
        auto_recalibrate: bool = False,
        max_age_hours: float | None = None,
    ):
        self._store = store
        self._poll_interval = poll_interval_s
        self._alert_fn = alert_fn or (lambda q, k: print(f"[Monitor] STALE: {q}/{k}"))
        self._recalibrate_fn = recalibrate_fn
        self._auto_recalibrate = auto_recalibrate
        self._max_age_hours = max_age_hours
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._watched: dict[str, list[str]] = {}  # {qubit: [keys]}

    # ── Watch list management ─────────────────────────────────────────────────

    def watch(self, qubit: str, keys: list[str] | None = None):
        """
        Add *qubit* (and optionally specific *keys*) to the watch list.

        If *keys* is None, all keys currently in the store are watched.
        """
        if keys is None:
            keys = self._store.all_keys(qubit)
        self._watched[qubit] = list(set(self._watched.get(qubit, []) + keys))

    def unwatch(self, qubit: str, keys: list[str] | None = None):
        """Remove *qubit* (or specific *keys*) from the watch list."""
        if keys is None:
            self._watched.pop(qubit, None)
        else:
            existing = self._watched.get(qubit, [])
            self._watched[qubit] = [k for k in existing if k not in keys]

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CalibMonitor")
        self._thread.start()
        print(f"[CalibrationMonitor] started (poll every {self._poll_interval:.0f} s)")

    def stop(self, timeout: float = 5.0):
        """Signal the thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        print("[CalibrationMonitor] stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_event.wait(self._poll_interval):
            self._check_once()

    def _check_once(self):
        for qubit, keys in self._watched.items():
            dynamic_keys = keys if keys else self._store.all_keys(qubit)
            for key in dynamic_keys:
                if self._store.is_stale(qubit, key, self._max_age_hours):
                    self._alert_fn(qubit, key)
                    if self._auto_recalibrate and self._recalibrate_fn is not None:
                        try:
                            self._recalibrate_fn(qubit, key)
                        except Exception as exc:
                            print(f"[CalibrationMonitor] recalibrate failed for {qubit}/{key}: {exc}")

    def check_now(self) -> list[tuple[str, str]]:
        """Perform a synchronous check and return ``[(qubit, key), ...]`` for stale entries."""
        stale = []
        for qubit, keys in self._watched.items():
            dynamic_keys = keys if keys else self._store.all_keys(qubit)
            for key in dynamic_keys:
                if self._store.is_stale(qubit, key, self._max_age_hours):
                    stale.append((qubit, key))
        return stale

    def __repr__(self) -> str:
        status = "running" if self.running else "stopped"
        return (
            f"CalibrationMonitor({status}, poll={self._poll_interval:.0f}s, "
            f"watching={list(self._watched.keys())})"
        )
