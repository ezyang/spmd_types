"""
Shared state for SPMD type checking.

This module holds thread-local state that needs to be shared between _scalar
and _checker without creating circular dependencies.

The state is per-thread because PyTorch's TorchFunctionMode stack is backed by
C++ thread-local storage (PythonTorchFunctionTLS), so modes are per-thread.
"""

import threading

_tls = threading.local()


def is_type_checking() -> bool:
    """Return True if type checking is active and not paused on this thread."""
    mode = getattr(_tls, "active_mode", None)
    return mode is not None and not mode._disabled


def _set_active_mode(mode):
    """Called by ``typecheck()`` after entering the TorchFunctionMode."""
    _tls.active_mode = mode


def _clear_active_mode():
    """Called by ``typecheck()`` before exiting the TorchFunctionMode."""
    _tls.active_mode = None


def is_strict() -> bool:
    """Return True if type checking is active and in strict mode."""
    mode = getattr(_tls, "active_mode", None)
    return mode is not None and mode._strict
