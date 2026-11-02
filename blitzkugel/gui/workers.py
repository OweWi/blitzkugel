"""Background execution of long-running tasks in a QThread."""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from ..core.progress import CancelCallback, CancelledError, ProgressCallback

log = logging.getLogger(__name__)

#: Signature of a task: ``task(progress, cancel) -> result``.
Task = Callable[[ProgressCallback, CancelCallback], Any]


class TaskWorker(QObject):
    """Runs a :data:`Task` in a worker thread and reports via Qt signals.

    Signals are delivered to the GUI thread through queued connections, so
    slots connected to them may touch widgets.
    """

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str, str)  # short message, traceback
    cancelled = pyqtSignal()

    def __init__(self, task: Task) -> None:
        super().__init__()
        self._task = task
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        """Ask the task to stop at the next progress update."""
        self._cancel.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._task(self.progress.emit, self._cancel.is_set)
        except CancelledError:
            log.info("Aufgabe abgebrochen")
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 – everything is reported to the user
            log.exception("Aufgabe fehlgeschlagen")
            self.failed.emit(str(exc) or type(exc).__name__, traceback.format_exc())
        else:
            self.finished.emit(result)


class TaskRunner(QObject):
    """Owns one worker thread at a time and wires up the callbacks."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(
        self,
        task: Task,
        on_finished: Callable[[Any], None],
        on_progress: Callable[[float, str], None],
        on_failed: Callable[[str, str], None],
        on_cancelled: Callable[[], None],
    ) -> None:
        """Start ``task`` in a new thread; only one task can run at a time."""
        if self.busy:
            raise RuntimeError("Es läuft bereits eine Berechnung.")
        thread = QThread(self)
        worker = TaskWorker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        for sig in (worker.finished, worker.failed, worker.cancelled):
            sig.connect(thread.quit)
        thread.finished.connect(self._cleanup)
        self._thread, self._worker = thread, worker
        thread.start()

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()

    def wait(self, msecs: int = 5000) -> None:
        """Cancel and wait for the thread (used when closing the window)."""
        if self._thread is not None:
            self.cancel()
            self._thread.wait(msecs)

    @pyqtSlot()
    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
