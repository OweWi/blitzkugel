"""Progress reporting and cooperative cancellation for long computations."""

from __future__ import annotations

from typing import Callable

#: Called with a fraction in [0, 1] and a short status text.
ProgressCallback = Callable[[float, str], None]
#: Returns True when the computation should stop as soon as possible.
CancelCallback = Callable[[], bool]


class CancelledError(Exception):
    """Raised inside a computation when the user requested cancellation."""


def no_progress(_fraction: float, _message: str) -> None:
    """Default progress callback that does nothing."""


def never_cancel() -> bool:
    """Default cancel callback."""
    return False


class ProgressScope:
    """Maps the progress of a sub-task into a range of the overall progress.

    Example: ``ProgressScope(cb, 0.2, 0.6)`` turns a sub-task's 0..1 into 0.2..0.6.
    """

    def __init__(
        self,
        callback: ProgressCallback,
        start: float,
        end: float,
        cancel: CancelCallback = never_cancel,
    ) -> None:
        self._callback = callback
        self._start = start
        self._end = end
        self._cancel = cancel

    def __call__(self, fraction: float, message: str) -> None:
        if self._cancel():
            raise CancelledError()
        f = min(max(fraction, 0.0), 1.0)
        self._callback(self._start + (self._end - self._start) * f, message)

    def sub(self, start: float, end: float) -> ProgressScope:
        """Create a nested scope covering ``start..end`` of this scope."""
        span = self._end - self._start
        return ProgressScope(
            self._callback, self._start + span * start, self._start + span * end, self._cancel
        )
