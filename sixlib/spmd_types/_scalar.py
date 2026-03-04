"""
Scalar wrapper with local SPMD type annotation.

Allows users to annotate Python scalars with an exact SPMD type, so that the
type checker uses the declared type instead of the weak _Scalar sentinel.

Example::

    from sixlib.spmd_types import Scalar, V, P

    # A rank-dependent scalar is Varying:
    typed_rank = Scalar(rank / world_size, {"tp": V})

    # A partial sum across ranks:
    typed_sum = Scalar(local_sum, {"tp": P})

    # Explicitly replicate (same as untyped, but opt-in):
    typed_const = Scalar(2.0, {"tp": R})
"""

from __future__ import annotations

from sixlib.spmd_types.types import (
    LocalSpmdType,
    PerMeshAxisLocalSpmdType,
)


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _unwrap_scalar(val: object):
    """If val is a Scalar, return its raw value; otherwise return val unchanged."""
    if isinstance(val, Scalar):
        return val.value
    return val


def _unwrap_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Replace each Scalar in args/kwargs with its raw value."""
    new_args = tuple(_unwrap_scalar(a) for a in args)
    new_kwargs = {k: _unwrap_scalar(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


class Scalar:
    """A Python scalar annotated with a local SPMD type.

    Wraps a numeric value (int, float, or complex -- not bool) together with
    an explicit SPMD type dict so that the type checker treats it as a typed
    operand rather than the weak ``_Scalar`` sentinel.

    Args:
        value: A numeric Python scalar.
        spmd_type: A ``LocalSpmdType`` dict mapping mesh axes to per-axis SPMD
            types (R, I, V, or P).
    """

    def __init__(self, value: int | float | complex, spmd_type: LocalSpmdType):
        if not _is_numeric_scalar(value):
            raise TypeError(
                f"Scalar value must be int, float, or complex (not bool), "
                f"got {type(value).__name__}: {value!r}"
            )
        for axis, typ in spmd_type.items():
            if not isinstance(typ, PerMeshAxisLocalSpmdType):
                raise TypeError(
                    f"Scalar type values must be PerMeshAxisLocalSpmdType "
                    f"(R, I, V, or P), got {typ!r} on axis {axis!r}"
                )
        self._local_type: LocalSpmdType = dict(spmd_type)
        self._value = value

    @property
    def value(self) -> int | float | complex:
        """The raw numeric value."""
        return self._value

    def __repr__(self) -> str:
        return f"Scalar({self._value!r}, {self._local_type!r})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Import from _state to avoid circular dependency (_checker imports _scalar).
        from sixlib.spmd_types._state import is_type_checking

        if is_type_checking():
            # SpmdTypeMode is active -- let the mode handle unwrapping
            return NotImplemented
        # No type checking active: unwrap Scalars to raw values and call func
        new_args, new_kwargs = _unwrap_args(args, kwargs)
        return func(*new_args, **new_kwargs)
