"""Low-level helpers for reading/writing the SPMD type attribute on tensors.

This module is intentionally minimal so that both ``_checker`` and
``_raw_dist`` can depend on it without creating a cycle.
"""

from __future__ import annotations

import torch
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    LocalSpmdType,
    PerMeshAxisLocalSpmdType,
)

# Attribute name for storing SPMD types on tensors.
#
# The attribute holds one of three states:
#   - absent:   truly untyped (created outside SpmdTypeMode)
#   - _FACTORY: factory sentinel (created by a factory op, real type TBD)
#   - dict:     real LocalSpmdType
_LOCAL_TYPE_ATTR = "_local_type"


class _FactorySentinel:
    """Sentinel stored in ``_LOCAL_TYPE_ATTR`` for factory tensors.

    A factory tensor is one produced by an op with no typed tensor inputs
    (e.g., ``torch.zeros``).  Its real SPMD type is deferred until the
    tensor is combined with a real-typed tensor or annotated via
    ``assert_type()``.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Factory"


_FACTORY = _FactorySentinel()


def has_local_type(value: object) -> bool:
    """Return True if the tensor (or Scalar) has a real SPMD type annotation.

    Returns False for factory tensors (use ``is_factory`` to check those).
    """
    attr = getattr(value, _LOCAL_TYPE_ATTR, None)
    return attr is not None and attr is not _FACTORY


def is_factory(value: object) -> bool:
    """Return True if the value is a factory tensor (type TBD)."""
    return getattr(value, _LOCAL_TYPE_ATTR, None) is _FACTORY


def set_factory(tensor: torch.Tensor) -> torch.Tensor:
    """Mark a tensor as factory. Returns the tensor for chaining."""
    setattr(tensor, _LOCAL_TYPE_ATTR, _FACTORY)
    return tensor


def get_local_type(value: object) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor or Scalar.

    Raises:
        AttributeError: If the object has no SPMD type annotations (or is
            factory).  Use ``has_local_type`` to check first.
    """
    result = getattr(value, _LOCAL_TYPE_ATTR, None)
    if result is None or result is _FACTORY:
        raise AttributeError(
            "Tensor has no SPMD type annotations. "
            "Use has_local_type() to check, or assert_type() to annotate."
        )
    return result


def set_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Set SPMD type on a tensor (internal). Returns the tensor for chaining.

    The caller is responsible for validating ``type`` before calling this.
    Overwrites the factory sentinel if present.
    """
    setattr(tensor, _LOCAL_TYPE_ATTR, type)
    return tensor


def get_axis_local_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisLocalSpmdType:
    """Get the SPMD type for a specific mesh axis.

    Raises:
        ValueError: If the tensor has no SPMD type annotations, or if the
            axis is not present in the tensor's SPMD type dict.

    Args:
        tensor: The tensor to query.
        axis: The mesh axis to look up (string name or ProcessGroup).
    """
    if not has_local_type(tensor):
        raise ValueError(
            "get_axis_local_type: tensor has no SPMD type annotations. "
            "Use has_local_type() to check first, or assert_type() to annotate."
        )
    local_type = get_local_type(tensor)
    if axis not in local_type:
        raise ValueError(
            f"Axis {format_axis(axis)} not found in tensor's SPMD type. "
            f"Tensor has axes: {set(local_type.keys())}"
        )
    return local_type[axis]
