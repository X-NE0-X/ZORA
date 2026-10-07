"""Cooperative cancellation scoped to one research worker, never other sessions."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class RunCancelled(BaseException):
    """A requested stop, not a transient provider error to retry."""


class RunControl:
    def __init__(self):
        self.cancelled = Event()

    def cancel(self) -> None:
        self.cancelled.set()


_CONTROL: ContextVar[RunControl | None] = ContextVar("zora_run_control", default=None)


@contextmanager
def cancellation_scope(control: RunControl):
    token = _CONTROL.set(control)
    try:
        check_cancelled()
        yield control
    finally:
        _CONTROL.reset(token)


def current_control() -> RunControl | None:
    return _CONTROL.get()


def check_cancelled() -> None:
    control = current_control()
    if control is not None and control.cancelled.is_set():
        raise RunCancelled("run cancelled by user")


def cancellable_sleep(seconds: float) -> None:
    control = current_control()
    if control is None:
        import time
        time.sleep(seconds)
    else:
        control.cancelled.wait(seconds)
        check_cancelled()
