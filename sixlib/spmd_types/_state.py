"""
Shared state for SPMD type checking.

This module holds global state that needs to be shared between _scalar and
_checker without creating circular dependencies.
"""

_spmd_type_mode_active: int = 0


def is_type_checking() -> bool:
    """Return True if a SpmdTypeMode is currently active."""
    return _spmd_type_mode_active > 0
