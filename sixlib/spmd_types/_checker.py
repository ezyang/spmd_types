"""
SPMD type checking logic, tensor type tracking, and global SPMD support.

This module provides:
- Functions to track SPMD types on tensors (local and global)
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
- Global SPMD shard propagation via DTensor's ShardingPropagator
- local_map for transitioning between local and global SPMD regions
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable

import torch
import torch.overrides
from sixlib.spmd_types._collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
)
from sixlib.spmd_types._local import convert, reinterpret
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    GlobalSpmdType,
    I,
    LocalSpmdType,
    P,
    PartitionSpec,
    PerMeshAxisGlobalSpmdType,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    R,
    Shard,
    SpmdTypeError,
    to_local_type,
    V,
)

log = logging.getLogger(__name__)

# =============================================================================
# Global SPMD Mode Tracking
# =============================================================================

_tls = threading.local()


def _is_global_mode() -> bool:
    return getattr(_tls, "global_mode", False)


def _set_global_mode(value: bool) -> None:
    _tls.global_mode = value


def _get_mesh_dim_sizes() -> dict[str, int] | None:
    return getattr(_tls, "mesh_dim_sizes", None)


def _set_mesh_dim_sizes(value: dict[str, int] | None) -> None:
    _tls.mesh_dim_sizes = value


def _get_axis_dim_size(axis: str) -> int | None:
    """Get the mesh dimension size for a specific axis.

    Checks thread-local mesh_dim_sizes first, then falls back to the
    process group for the axis.
    """
    sizes = _get_mesh_dim_sizes()
    if sizes is not None and axis in sizes:
        return sizes[axis]
    # Fall back to PG for the axis if available.
    from sixlib.spmd_types._mesh import get_mesh
    import torch.distributed as dist
    mesh = get_mesh()
    if mesh is not None and dist.is_initialized():
        return dist.get_world_size(mesh.get_group(axis))


# =============================================================================
# Fix Suggestion Engine
# =============================================================================

_TYPE_FULL_NAMES = {
    R: "Replicate",
    I: "Invariant",
    V: "Varying",
    P: "Partial",
}

# Each entry: (from_type, to_type_instance, operation_str, consequence_str)
# When a type error occurs, we try replacing one operand's type and re-running
# inference. If it succeeds, we report the fix.
_FIX_CANDIDATES: list[
    tuple[PerMeshAxisLocalSpmdType, PerMeshAxisLocalSpmdType, str, str]
] = [
    (
        I,
        R,
        "reinterpret(tensor, {axis_arg}, src=I, dst=R)",
        "no-op forward, all-reduce in backward",
    ),
    (
        I,
        V,
        "reinterpret(tensor, {axis_arg}, src=I, dst=V)",
        "no-op forward, all-reduce in backward",
    ),
    (
        I,
        P,
        "convert(tensor, {axis_arg}, src=I, dst=P)",
        "zeros non-rank-0 in forward, no-op backward",
    ),
    (
        P,
        R,
        "all_reduce(tensor, {axis_arg}, src=P, dst=R)",
        "all-reduce in forward, all-reduce in backward",
    ),
    (
        P,
        I,
        "all_reduce(tensor, {axis_arg}, src=P, dst=I)",
        "all-reduce in forward, no-op backward",
    ),
    (
        R,
        P,
        "convert(tensor, {axis_arg}, src=R, dst=P)",
        "zeros non-rank-0 in forward, zeros non-rank-0 in backward",
    ),
]
# NB: We don't suggest R->I because compute typically happens on R, not I.

# NB: We don't suggest
# (Partial, V, "reduce_scatter(tensor, axis, src=P, dst=V)", ...)
# because this requires reasoning about the desired size of the output; if the
# original code is the correct size, the rewrite is incorrect.

# NB: We don't suggest
# (Varying, P, "reinterpret(tensor, axis, src=V, dst=P)", ...)
# because the operation requiring P should have just been fed V directly.


def _suggest_fixes(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> list[tuple[str, str, PerMeshAxisLocalSpmdType]]:
    """Try each candidate fix and return suggestions for ones that work.

    For each candidate (from_type, to_type, ...):
    1. Check if from_type exists in axis_types
    2. Replace one occurrence with to_type
    3. Call infer_fn on the modified list
    4. If no exception, this is a valid fix -- include it

    The ``infer_fn`` must be a *raw* inference function that raises plain
    ``SpmdTypeError`` without calling ``_format_error_with_suggestions``.
    This makes recursion structurally impossible.

    Args:
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function that takes (axis, axis_types) and
            returns the inferred output type, or raises ``SpmdTypeError``.

    Returns:
        List of (operation_str, consequence_str, from_type) tuples.
    """
    # Compute the axis argument text once
    if isinstance(axis, str):
        axis_arg = repr(axis)  # e.g. "'tp'"
    else:
        axis_arg = "pg"

    suggestions: list[tuple[str, str, PerMeshAxisLocalSpmdType]] = []
    for from_type, to_type, operation_template, consequence in _FIX_CANDIDATES:
        # Find the first operand matching from_type
        idx = None
        for i, t in enumerate(axis_types):
            if t is from_type:
                idx = i
                break
        if idx is None:
            continue
        # Try replacing that operand
        modified = list(axis_types)
        modified[idx] = to_type
        try:
            fix_output = infer_fn(axis, modified)
        except SpmdTypeError:
            continue
        # Filter: does this fix preserve the natural output of the other operands?
        remaining = [t for i, t in enumerate(axis_types) if i != idx]
        if remaining:
            try:
                natural_output = infer_fn(axis, remaining)
                if fix_output != natural_output:
                    continue  # Fix changes the output type -- skip
            except (SpmdTypeError, ValueError):
                pass  # Can't determine natural output -- keep the suggestion
        operation = operation_template.format(axis_arg=axis_arg)
        suggestions.append((operation, consequence, from_type))
    return suggestions


def _format_error_with_suggestions(
    base_msg: str,
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [DeviceMeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> str:
    """Format an error message, appending fix suggestions if any exist.

    Args:
        base_msg: The base error message to display.
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function used to discover valid fixes.
    """
    suggestions = _suggest_fixes(axis, axis_types, infer_fn)
    if not suggestions:
        return base_msg
    lines = [base_msg, "Are you missing a collective? e.g.,"]
    for operation, consequence, from_type in suggestions:
        lines.append(
            f"  {operation} on the {_TYPE_FULL_NAMES[from_type]} operand ({consequence})"
        )
    return "\n".join(lines)


# =============================================================================
# SPMD Type Tracking on Tensors
# =============================================================================

# Attribute name for storing SPMD types on tensors.
# In local mode this holds a LocalSpmdType, in global mode a GlobalSpmdType.
_SPMD_TYPE_ATTR = "_spmd_type"


def _validate_and_canonicalize_local(type: LocalSpmdType) -> LocalSpmdType:
    """Validate and canonicalize a local SPMD type.

    Validates that all values are valid local SPMD types (R, I, V, or P).
    Shard types are not valid local SPMD types -- they are used as arguments to
    collective operations but must not be stored on tensors.

    Canonicalizes by removing V entries: omitted mesh axes default to Varying,
    so explicit V entries are redundant.  Stripping them ensures that
    ``{"tp": R, "dp": V}`` and ``{"tp": R}`` compare as equal.

    Args:
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P).
    """
    for axis, typ in type.items():
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            if isinstance(typ, Shard):
                raise TypeError(
                    f"Shard type {typ!r} on axis {format_axis(axis)} cannot be stored "
                    f"as a local SPMD type. Shard is only valid as src/dst in "
                    f"collective operations. Use V instead for local type tracking."
                )
            raise TypeError(
                f"Expected PerMeshAxisLocalSpmdType (R, I, V, or P) on axis "
                f"{format_axis(axis)}, got {typ!r}"
            )
    return {axis: typ for axis, typ in type.items() if typ is not V}


def _validate_and_canonicalize_global(type: GlobalSpmdType) -> GlobalSpmdType:
    """Validate and canonicalize a global SPMD type.

    Valid types: R, I, P, S(i). V is rejected (global SPMD requires an explicit
    type on every axis).

    No canonicalization needed: S(i) entries are stored as-is, and V is rejected
    rather than stripped.
    """
    for axis, typ in type.items():
        if isinstance(typ, Shard):
            continue
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            raise SpmdTypeError(
                f"Expected R, I, P, or S(i) on axis "
                f"{format_axis(axis)}, got {typ!r}"
            )
        if typ is V:
            raise SpmdTypeError(
                f"Varying (V) is not a valid type in global SPMD on axis "
                f"{format_axis(axis)}. Use R, I, P, or S(i) instead."
            )
    return dict(type)


def has_local_type(tensor: torch.Tensor) -> bool:
    """Return True if the tensor has SPMD type annotations.

    Args:
        tensor: The tensor to check for SPMD type annotations.
    """
    return hasattr(tensor, _SPMD_TYPE_ATTR)


def get_local_type(tensor: torch.Tensor) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor.

    In global mode, the returned dict may contain Shard values (GlobalSpmdType).

    Args:
        tensor: The tensor to retrieve SPMD type annotations from.

    Raises:
        AttributeError: If the tensor has no SPMD type annotations.
            Use ``has_local_type`` to check first.
    """
    result = getattr(tensor, _SPMD_TYPE_ATTR, None)
    if result is None:
        raise AttributeError(
            "Tensor has no SPMD type annotations. "
            "Use has_local_type() to check, or assert_type() to annotate."
        )
    return result


def _set_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Set SPMD type on a tensor (internal). Returns the tensor for chaining.

    Args:
        tensor: The tensor to annotate with an SPMD type.
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.
    """
    validate = _validate_and_canonicalize_global if _is_global_mode() else _validate_and_canonicalize_local
    setattr(tensor, _SPMD_TYPE_ATTR, validate(type))
    return tensor


def assert_type(  # noqa: C901
    tensor: torch.Tensor,
    type: LocalSpmdType,
    partition_spec: PartitionSpec | None = None,
) -> torch.Tensor:
    """
    Assert or set the SPMD type on a tensor.

    If the tensor has no SPMD type, sets it.
    If the tensor already has an SPMD type, checks it equals the provided type.
    Returns the tensor for chaining.

    **Global SPMD mode** (``SpmdTypeMode(global_mode=True)``):
      Valid types are R, I, P, and S(i). V is rejected.
      S(i) is stored directly as a first-class type.
      ``partition_spec`` is not supported.

    **Local SPMD mode** (default):
      Valid types are R, I, V, and P. Omitted axes default to V.
      S(i) can be used as syntax sugar: it is equivalent to V on that axis
      plus a ``partition_spec`` entry. S(i) cannot be mixed with an explicit
      ``partition_spec``, and two axes cannot shard the same tensor dimension
      via S(i).

    Args:
        tensor: The tensor to assert or set SPMD type on.
        type: Dict mapping mesh axes to per-axis SPMD types.
        partition_spec: Optional PartitionSpec (local mode only).

    Raises:
        TypeError: If type contains invalid values, V is used in global mode,
            partition_spec is used in global mode, or S(i) is mixed with
            partition_spec in local mode.
        SpmdTypeError: If partition_spec length doesn't match tensor ndim, or
            if a partition_spec axis conflicts with a non-Varying type.
        AssertionError: If existing type doesn't match provided type.
    """
    ############ Compute canonical type ############
    if _is_global_mode():
        if partition_spec is not None:
            raise TypeError(
                "partition_spec is not supported in global SPMD mode. "
                "Use S(i) in type instead."
            )
        canonical = _validate_and_canonicalize_global(type)
    else:
        # Separate S(i) entries from R/I/V/P entries. In local mode, S(i) is
        # syntax sugar: the axis becomes V and sharding info is optional metadata.
        local_type: LocalSpmdType = {}
        shard_entries: dict[DeviceMeshAxis, Shard] = {}
        for axis, typ in type.items():
            if isinstance(typ, Shard):
                shard_entries[axis] = typ
            else:
                local_type[axis] = typ

        if shard_entries:
            if partition_spec is not None:
                raise TypeError(
                    "Cannot use S(i) in type and an explicit partition_spec at the "
                    "same time. Use one or the other."
                )
            dim_to_axes: dict[int, DeviceMeshAxis] = {}
            for axis, shard in shard_entries.items():
                dim = shard.dim
                if tensor.ndim == 0 or dim < -tensor.ndim or dim >= tensor.ndim:
                    raise SpmdTypeError(
                        f"S({dim}) on axis {format_axis(axis)} is out of bounds "
                        f"for tensor with {tensor.ndim} dimensions"
                    )
                resolved_dim = dim % tensor.ndim if dim < 0 else dim
                if resolved_dim in dim_to_axes:
                    raise SpmdTypeError(
                        f"Multiple mesh axes shard the same tensor dimension "
                        f"{resolved_dim}: {format_axis(dim_to_axes[resolved_dim])} "
                        f"and {format_axis(axis)}. Use an explicit PartitionSpec "
                        f"to specify the sharding order."
                    )
                dim_to_axes[resolved_dim] = axis

        if partition_spec is not None:
            if len(partition_spec) != tensor.ndim:
                raise SpmdTypeError(
                    f"PartitionSpec length {len(partition_spec)} doesn't match "
                    f"tensor ndim {tensor.ndim}"
                )
            for dim_entry in partition_spec:
                if dim_entry is None:
                    continue
                axes = (dim_entry,) if isinstance(dim_entry, str) else dim_entry
                for axis in axes:
                    axis_type = local_type.get(axis)
                    if axis_type is not None and axis_type is not V:
                        raise SpmdTypeError(
                            f"Mesh axis {format_axis(axis)} appears in "
                            f"partition_spec (implying Varying/Shard) but is "
                            f"specified as {axis_type} in type."
                        )

        canonical = _validate_and_canonicalize_local(local_type)

    ############ Set or check ############
    if not has_local_type(tensor):
        return _set_local_type(tensor, canonical)
    existing = get_local_type(tensor)
    if existing != canonical:
        raise AssertionError(
            f"SPMD type mismatch: tensor has {existing}, expected {canonical}"
        )
    return tensor


def assert_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Deprecated: use ``assert_type`` instead."""
    return assert_type(tensor, type)


def get_axis_local_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisLocalSpmdType:
    """Get the SPMD type for a specific mesh axis.

    Returns V (Varying) if the axis is not explicitly stored, since omitted
    axes default to Varying.

    Raises:
        ValueError: If the tensor has no SPMD type annotations.

    Args:
        tensor: The tensor to query.
        axis: The mesh axis to look up (string name or ProcessGroup).
    """
    if not has_local_type(tensor):
        raise ValueError(
            "get_axis_local_type: tensor has no SPMD type annotations. "
            "Use has_local_type() to check first, or assert_type() to annotate."
        )
    return get_local_type(tensor).get(axis, V)


# =============================================================================
# Type Inference Logic
# =============================================================================


class OpLinearity(Enum):
    """Classifies how a torch op interacts with Partial (P) types.

    - NONLINEAR: P cannot propagate (safe default for unclassified ops).
    - LINEAR: Linear map on direct sum; all-P -> P.
      Examples: addition, subtraction, concat, clone.
    - MULTILINEAR: Linear in each factor separately; P in one factor with R
      in others -> P, but P in multiple factors is forbidden.
      Examples: multiplication, matmul, einsum.
    """

    NONLINEAR = auto()
    LINEAR = auto()
    MULTILINEAR = auto()


def _infer_local_type_for_axis_raw(  # noqa: C901
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """Raw inference logic -- raises plain ``SpmdTypeError`` without suggestions.

    The public wrapper ``infer_local_type_for_axis`` catches these errors and
    enriches them with fix suggestions.

    Args:
        axis: The mesh axis name (used for error messages).
        axis_types: List of input SPMD types for this axis.
        out_partial: If True, reinterpret the inferred result as Partial.
        linearity: How the op interacts with Partial types.
    """
    if not axis_types:
        if out_partial:
            return P
        raise ValueError(f"No types provided for axis {format_axis(axis)}")

    # Check type compatibility and infer output type
    type_set = set(axis_types)

    if len(type_set) == 1:
        inferred_type = axis_types[0]
        if inferred_type is P:
            if linearity is OpLinearity.NONLINEAR:
                raise SpmdTypeError(
                    f"Partial type on axis {format_axis(axis)} cannot propagate through "
                    f"non-linear ops. Use all_reduce or reduce_scatter first. "
                    f"Found types: {axis_types}"
                )
            elif linearity is OpLinearity.MULTILINEAR:
                p_count = sum(1 for t in axis_types if t is P)
                if p_count > 1:
                    raise SpmdTypeError(
                        f"Partial in multiple factors of multilinear op on axis "
                        f"{format_axis(axis)} is forbidden. "
                        f"Found types: {axis_types}"
                    )
                # Single P -> P (unary multilinear is fine)
            # LINEAR: all-P -> P, pass through
    elif type_set == {R, V}:
        # Mixed replicate/varying -> varying
        inferred_type = V
    elif I in type_set and len(type_set) > 1:
        raise SpmdTypeError(
            f"Invariant type on axis {format_axis(axis)} cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif P in type_set:
        # P mixed with non-P types
        if linearity is OpLinearity.MULTILINEAR:
            p_count = sum(1 for t in axis_types if t is P)
            non_p = type_set - {P}
            if p_count > 1:
                raise SpmdTypeError(
                    f"Partial in multiple factors of multilinear op on axis "
                    f"{format_axis(axis)} is forbidden. "
                    f"Found types: {axis_types}"
                )
            if non_p == {R}:
                inferred_type = P  # P in one factor, R in others -> P
            else:
                raise SpmdTypeError(
                    f"Partial type on axis {format_axis(axis)} can only multiply with "
                    f"Replicate. Found types: {axis_types}"
                )
        else:
            # NONLINEAR and LINEAR both reject P mixed with non-P
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} can only combine with partial. "
                f"Found types: {axis_types}"
            )
    else:
        raise SpmdTypeError(
            f"Incompatible types on axis {format_axis(axis)}: {axis_types}"
        )

    # Apply out_partial: reinterpret as P
    if out_partial:
        if inferred_type is V or inferred_type is P:
            return P
        elif inferred_type is R:
            raise SpmdTypeError(
                f"out_partial_axes includes axis {format_axis(axis)} but inferred type "
                f"is R (Replicate). A replicated result cannot be partial -- this likely "
                f"indicates an unsharded contraction dimension. "
                f"Input types: {axis_types}"
            )
        else:
            raise SpmdTypeError(
                f"Cannot mark axis {format_axis(axis)} as partial with type {inferred_type}. "
                f"out_partial_axes is only valid for V or P types."
            )

    return inferred_type


def infer_local_type_for_axis(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """
    Infer the output SPMD type for a single mesh axis given input types.

    Args:
        axis: The mesh axis name (for error messages)
        axis_types: List of input types for this axis
        out_partial: If True, reinterpret the result as Partial
        linearity: How the op interacts with Partial types

    Returns:
        The inferred output type

    Raises:
        SpmdTypeError: If the input types are incompatible
    """
    try:
        return _infer_local_type_for_axis_raw(axis, axis_types, out_partial, linearity)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            _format_error_with_suggestions(
                str(e),
                axis,
                axis_types,
                lambda a, t: _infer_local_type_for_axis_raw(
                    a, t, out_partial, linearity
                ),
            )
        ) from None


def infer_output_type(
    input_types_list: list[LocalSpmdType],
    out_partial_axes: set[DeviceMeshAxis] | None = None,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> LocalSpmdType:
    """
    Infer output SPMD types from a list of input types.

    This implements the typing rules for operations like einsum:
    - If all operands are R -> output is R
    - If all operands are I -> output is I
    - If all operands are V -> output is V
    - If all operands are P -> output is P (linear ops only)
    - Mixed R/V -> output is V
    - I cannot mix with other types
    - P cannot mix with non-P types

    Args:
        input_types_list: List of LocalSpmdType dicts, one per operand
        out_partial_axes: Optional set of mesh axis names to mark as partial
        linearity: How the op interacts with Partial types

    Returns:
        LocalSpmdType dict for the output
    """
    if out_partial_axes is None:
        out_partial_axes = set()

    # Collect all mesh axes mentioned
    all_axes: set[DeviceMeshAxis] = set()
    for typ in input_types_list:
        all_axes.update(typ.keys())
    all_axes.update(out_partial_axes)

    # Infer output type for each axis
    output_type: LocalSpmdType = {}
    for axis in all_axes:
        axis_types = []
        for typ in input_types_list:
            axis_types.append(typ.get(axis, V))

        output_type[axis] = infer_local_type_for_axis(
            axis, axis_types, out_partial=axis in out_partial_axes, linearity=linearity
        )

    return output_type


def _linear(*tys: LocalSpmdType) -> LocalSpmdType:
    """Type of a linear combination (addition)."""
    return infer_output_type(list(tys), linearity=OpLinearity.LINEAR)


def _multilinear(*tys: LocalSpmdType) -> LocalSpmdType:
    """Type of a multilinear product (matmul, elementwise mul)."""
    return infer_output_type(list(tys), linearity=OpLinearity.MULTILINEAR)


# =============================================================================
# SPMD Function Registry
# =============================================================================

# Default src/dst for each SPMD collective/local op.  When a kwarg is omitted
# by the caller the Python default from the function signature applies; we
# record those defaults here so that __torch_function__ can recover them
# (handle_torch_function does not forward defaults).
# A value of ``None`` means the parameter is required (no default).


@dataclass(frozen=True)
class _OpSpec:
    """Specification of how a torch op interacts with SPMD types.

    Attributes:
        linearity: How the op interacts with Partial (P) types.
        tensor_args: Positional arg indices (0-based) that are tensor inputs.
            Each position may hold a single tensor OR a list of tensors.
            Only needed for LINEAR ops (scalars at these positions become R).
        tensor_kwargs: Kwarg names that are tensor inputs.
        tensor_varargs_from: If set, all positional args from this index onward
            are tensor inputs (for ops with *args like einsum).
    """

    linearity: OpLinearity
    tensor_args: tuple[int, ...] = ()
    tensor_kwargs: tuple[str, ...] = ()
    tensor_varargs_from: int | None = None
    fixed_args: tuple[int, ...] = ()


_OP_REGISTRY: dict[Callable, _OpSpec] = {
    # =================================================================
    # LINEAR -- binary arithmetic (add / sub)
    # =================================================================
    torch.add: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.Tensor.add: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.add_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__add__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__radd__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__iadd__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.sub: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.subtract: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.Tensor.sub: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.subtract: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.sub_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__sub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__rsub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__isub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    # =================================================================
    # LINEAR -- negation / positive
    # =================================================================
    torch.neg: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.neg: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.neg_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.__neg__: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.negative: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.negative: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.negative_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.positive: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.positive: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- clone / detach
    # =================================================================
    torch.clone: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.clone: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.detach: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.detach: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.detach_: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- reductions (sum, mean)
    # =================================================================
    torch.sum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.sum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.mean: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.mean: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.nansum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.nanmean: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- concat / stack (tensor list at pos 0)
    # =================================================================
    torch.cat: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.concat: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.concatenate: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.hstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.vstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.dstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.column_stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.row_stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.block_diag: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- structural / shape ops (tensor at pos 0)
    # =================================================================
    torch.reshape: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.reshape: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.view: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.view_as: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.transpose: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.transpose: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.transpose_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.t: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.t: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.t_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.permute: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.permute: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.contiguous: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.flatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unflatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unflatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.ravel: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.ravel: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.squeeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.squeeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.squeeze_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unsqueeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unsqueeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unsqueeze_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.expand: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.expand_as: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.broadcast_to: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.narrow: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.narrow: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.index_select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.index_select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.gather: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.gather: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.split: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.split: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.chunk: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.chunk: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unbind: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unbind: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flip: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.flip: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.fliplr: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flipud: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.roll: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.roll: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.movedim: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.movedim: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.moveaxis: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.moveaxis: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.swapaxes: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.swapaxes: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.swapdims: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.swapdims: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.diagonal: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.diagonal: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.repeat: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.tile: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.tile: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.repeat_interleave: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.repeat_interleave: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unfold: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR with fixed_args -- division (linear in numerator only)
    # =================================================================
    torch.div: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)),
    torch.divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)
    ),
    torch.true_divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)
    ),
    torch.Tensor.div: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)),
    torch.Tensor.divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.true_divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.div_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)),
    torch.Tensor.__truediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.__rtruediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(0,)
    ),
    torch.Tensor.__itruediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    # =================================================================
    # MULTILINEAR -- multiplication
    # =================================================================
    torch.mul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.multiply: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.multiply: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mul_: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__mul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__rmul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__imul__: _OpSpec(OpLinearity.MULTILINEAR),
    # =================================================================
    # MULTILINEAR -- matmul / mm / bmm
    # =================================================================
    torch.matmul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.mm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.bmm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.matmul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.bmm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__matmul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__rmatmul__: _OpSpec(OpLinearity.MULTILINEAR),
    # =================================================================
    # MULTILINEAR -- einsum, dot, mv, etc.
    # =================================================================
    torch.einsum: _OpSpec(OpLinearity.MULTILINEAR),
    torch.dot: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.dot: _OpSpec(OpLinearity.MULTILINEAR),
    torch.mv: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mv: _OpSpec(OpLinearity.MULTILINEAR),
}

# Type-level decompositions for compound ops.
# Each function mirrors the original op's signature (taking one LocalSpmdType
# per tensor arg) and returns the output type, using _linear/_multilinear to
# describe the algebraic structure.  These mirror PyTorch decompositions in
# fbcode/caffe2/torch/_decomp/decompositions.py but operate purely on types.


def _addmm_types(
    self_t: LocalSpmdType, mat1_t: LocalSpmdType, mat2_t: LocalSpmdType
) -> LocalSpmdType:
    # addmm(self, mat1, mat2) = self + mm(mat1, mat2)
    return _linear(_multilinear(mat1_t, mat2_t), self_t)


def _addmv_types(
    self_t: LocalSpmdType, mat_t: LocalSpmdType, vec_t: LocalSpmdType
) -> LocalSpmdType:
    # addmv(self, mat, vec) = self + mv(mat, vec)
    return _linear(_multilinear(mat_t, vec_t), self_t)


def _addbmm_types(
    self_t: LocalSpmdType, batch1_t: LocalSpmdType, batch2_t: LocalSpmdType
) -> LocalSpmdType:
    # addbmm(self, batch1, batch2) = self + bmm(batch1, batch2).sum(0)
    # sum is LINEAR so the intermediate doesn't change the type
    return _linear(_multilinear(batch1_t, batch2_t), self_t)


def _baddbmm_types(
    self_t: LocalSpmdType, batch1_t: LocalSpmdType, batch2_t: LocalSpmdType
) -> LocalSpmdType:
    # baddbmm(self, batch1, batch2) = self + bmm(batch1, batch2)
    return _linear(_multilinear(batch1_t, batch2_t), self_t)


def _addr_types(
    self_t: LocalSpmdType, vec1_t: LocalSpmdType, vec2_t: LocalSpmdType
) -> LocalSpmdType:
    # addr(self, vec1, vec2) = self + outer(vec1, vec2)
    return _linear(_multilinear(vec1_t, vec2_t), self_t)


_DECOMP_TYPE_RULES: dict[Callable, Callable[..., LocalSpmdType]] = {
    torch.addmm: _addmm_types,
    torch.Tensor.addmm: _addmm_types,
    torch.addmv: _addmv_types,
    torch.Tensor.addmv: _addmv_types,
    torch.addbmm: _addbmm_types,
    torch.Tensor.addbmm: _addbmm_types,
    torch.baddbmm: _baddbmm_types,
    torch.Tensor.baddbmm: _baddbmm_types,
    torch.addr: _addr_types,
    torch.Tensor.addr: _addr_types,
}


def _iter_tensor_args(args: tuple, kwargs: dict):
    """Yield all tensor arguments from args and kwargs.

    Flattens one level of list/tuple in both positional args and kwargs values
    (for ops like torch.cat/stack that accept tensor lists).

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor):
            yield arg
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    yield item
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            yield v
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    yield item


def _check_all_typed(args: tuple, kwargs: dict) -> None:
    """Raise ``SpmdTypeError`` if typed and untyped tensors are mixed.

    Called once at the top of the regular-op path in strict mode so that
    ``_collect_tensor_types`` itself stays simple and unconditional.

    Uses ``has_local_type`` rather than truthiness of the types dict because
    ``_validate_and_canonicalize_local`` strips V entries -- a tensor annotated as all-V stores
    ``{}`` but should still count as typed.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    has_typed = False
    has_untyped = False
    for t in _iter_tensor_args(args, kwargs):
        if has_local_type(t):
            has_typed = True
        else:
            has_untyped = True
        if has_typed and has_untyped:
            raise SpmdTypeError(
                "Strict mode: operation mixes tensors with SPMD type annotations "
                "and tensors without. All tensor operands must be annotated. "
                "Use SpmdTypeMode(strict=False) if you want partial type checking."
            )


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _collect_input_types(  # noqa: C901
    args: tuple, kwargs: dict, spec: _OpSpec | None = None
) -> list[LocalSpmdType]:
    """Collect SPMD types from tensor args and (for LINEAR ops) scalar args.

    For LINEAR ops with a spec, numeric scalars at declared tensor-input
    positions are included as Replicate on all known mesh axes.  For other ops
    (MULTILINEAR, NONLINEAR, or no spec), only tensor types are collected and
    scalars are ignored.

    TODO: A Python scalar is ambiguous between R and V.  We conservatively
    assume R for now; revisit later.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
        spec: Optional op specification for detecting scalar tensor positions.
    """
    result: list[LocalSpmdType] = []
    for t in _iter_tensor_args(args, kwargs):
        if has_local_type(t):
            result.append(get_local_type(t))

    if result and spec is not None and spec.linearity is OpLinearity.LINEAR:
        all_axes: set[DeviceMeshAxis] = set()
        for typ in result:
            all_axes.update(typ.keys())
        if all_axes:
            scalar_type: LocalSpmdType = {axis: R for axis in all_axes}

            def _check(val: object) -> None:
                if _is_numeric_scalar(val):
                    result.append(scalar_type)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if _is_numeric_scalar(item):
                            result.append(scalar_type)

            for i in spec.tensor_args:
                if i < len(args):
                    _check(args[i])
            if spec.tensor_varargs_from is not None:
                for i in range(spec.tensor_varargs_from, len(args)):
                    _check(args[i])
            for name in spec.tensor_kwargs:
                if name in kwargs:
                    _check(kwargs[name])

    return result


def _set_result_type(result: object, output_type: LocalSpmdType) -> None:
    """Set the same SPMD type on all result tensor(s).

    Handles single tensors, flat list/tuple of tensors, and arbitrary nested
    structures (e.g., NamedTuples from ops like torch.linalg.lu_factor).
    The common cases (single tensor, flat sequence) are checked first to
    avoid the overhead of pytree flattening.

    Args:
        result: The operation result -- a single tensor, a list/tuple of tensors,
            or an arbitrary nested structure containing tensors.
        output_type: The LocalSpmdType dict to set on each result tensor.
    """
    if isinstance(result, torch.Tensor):
        _set_local_type(result, output_type)
        return
    if isinstance(result, (list, tuple)):
        # Check if it's a flat sequence of tensors/non-tensors (common case).
        all_flat = True
        for item in result:
            if isinstance(item, torch.Tensor):
                _set_local_type(item, output_type)
            elif isinstance(item, (list, tuple, dict)):
                all_flat = False
        if all_flat:
            return
    # Fall back to pytree for nested structures.
    flat, _ = torch.utils._pytree.tree_flatten(result)
    for item in flat:
        if isinstance(item, torch.Tensor):
            _set_local_type(item, output_type)


def _set_result_types(
    result: object, output_types: object
) -> None:
    """Set per-output SPMD types on a multi-output result.

    Walks result and output_types in lockstep using recursive isinstance
    checks (following DTensor's OpDispatcher.wrap pattern). Preserves
    container types (tuple vs list).

    Args:
        result: The operation result -- a single tensor or a (possibly nested)
            list/tuple of tensors.
        output_types: A single LocalSpmdType dict (for a tensor) or a list/tuple
            of output_types matching the result structure.
    """
    if isinstance(result, torch.Tensor):
        if isinstance(output_types, dict):
            _set_local_type(result, output_types)
        return
    if isinstance(result, (list, tuple)):
        assert isinstance(output_types, (list, tuple))
        for item, typ in zip(result, output_types):
            _set_result_types(item, typ)


def _apply_fixed_args(  # noqa: C901
    func: Callable,
    args: tuple,
    kwargs: dict,
    spec: _OpSpec,
    input_types_list: list[LocalSpmdType],
) -> list[LocalSpmdType]:
    """Filter fixed_args for LINEAR ops when Partial is present.

    ``fixed_args`` lists positional arg indices that must be held fixed (not P)
    for the op to be linear in the remaining args.  For example, ``div(a, b)``
    is linear in ``a`` when ``b`` is fixed, so ``fixed_args=(1,)``.

    When P is present among the non-fixed tensor args:
    1. Validate that tensor args at fixed_args positions don't have P on any axis.
    2. Exclude their types from the returned list so LINEAR inference sees only
       the "free" args (and doesn't reject P + R mixing).

    When no P is present in the free args, return the original list unchanged
    so normal inference rules apply (e.g., R + V -> V).

    Args:
        func: The torch function being called.
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
        spec: The op specification (must have non-empty fixed_args).
        input_types_list: The already-collected input types list.
    """
    # Identify which input_types_list entries came from fixed_args positions.
    # We re-walk args to figure out which types are "fixed" vs "free".
    fixed_positions = set(spec.fixed_args)
    free_types: list[LocalSpmdType] = []
    fixed_types: list[LocalSpmdType] = []

    for i in spec.tensor_args:
        if i < len(args):
            val = args[i]
            if isinstance(val, torch.Tensor) and has_local_type(val):
                if i in fixed_positions:
                    fixed_types.append(get_local_type(val))
                else:
                    free_types.append(get_local_type(val))
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, torch.Tensor) and has_local_type(item):
                        if i in fixed_positions:
                            fixed_types.append(get_local_type(item))
                        else:
                            free_types.append(get_local_type(item))

    # Check if P is present in any free arg on any axis
    has_p_in_free = any(P in typ.values() for typ in free_types)
    # Also check default V (missing from dict means V, not P) -- V != P, so fine.

    if not has_p_in_free:
        # No P in free args -- normal inference, include all types.
        return input_types_list

    # P is present in free args -- validate fixed args don't have P
    for fixed_type in fixed_types:
        for axis, typ in fixed_type.items():
            if typ is P:
                raise SpmdTypeError(
                    f"{func.__name__}: Partial type in fixed argument "
                    f"(denominator/divisor) on axis {format_axis(axis)} is not allowed. "
                    f"Division is only linear in the numerator."
                )

    # Exclude fixed_types from input_types_list: return only free_types
    # plus any scalar types that were appended by _collect_input_types.
    # scalar_types count = len(input_types_list) - len(free_types) - len(fixed_types)
    n_tensor_types = len(free_types) + len(fixed_types)
    scalar_types = input_types_list[n_tensor_types:]
    return free_types + scalar_types


# Every function in this registry must accept (x: Tensor, axis, *, src=..., dst=...)
# as its leading arguments, since __torch_function__ recovers src/dst from
# args[0], args[1], and kwargs.
_SPMD_FUNCTION_DEFAULTS: dict[Callable, dict[str, PerMeshAxisSpmdType | None]] = {
    all_reduce: {"src": P, "dst": None},
    all_gather: {"src": V, "dst": None},
    reduce_scatter: {"src": P, "dst": V},
    all_to_all: {"src": V, "dst": V},
    redistribute: {"src": None, "dst": None},
    reinterpret: {"src": None, "dst": None},
    convert: {"src": None, "dst": None},
}


# =============================================================================
# Global SPMD Shard Propagation (via DTensor ShardingPropagator)
# =============================================================================

def _to_dt_placement(typ: PerMeshAxisSpmdType):
    """Convert our per-axis SPMD type to a DTensor Placement.

    I (Invariant) maps to Replicate since DTensor has no Invariant concept.
    The caller is responsible for recovering I in the output (R→I).
    """
    from torch.distributed.tensor.placement_types import (
        Partial as DTPartial,
        Replicate as DTReplicate,
        Shard as DTShard,
    )

    if isinstance(typ, Shard):
        return DTShard(typ.dim)
    if typ is R or typ is I:
        return DTReplicate()
    if typ is P:
        return DTPartial()
    raise ValueError(f"Cannot convert {typ} to DTensor placement")


def _from_dt_placement(placement) -> PerMeshAxisSpmdType:
    """Convert a DTensor Placement to our per-axis SPMD type."""
    from torch.distributed.tensor.placement_types import (
        Partial as DTPartial,
        Replicate as DTReplicate,
        Shard as DTShard,
    )

    if isinstance(placement, DTShard):
        return Shard(placement.dim)
    if isinstance(placement, DTReplicate):
        return R
    if isinstance(placement, DTPartial):
        return P
    raise ValueError(f"Cannot convert DTensor placement {placement}")


# Dunder names that don't match their aten op name after stripping __.
_DUNDER_NAME_OVERRIDES: dict[str, str] = {
    "truediv": "true_divide",
    "floordiv": "floor_divide",
}

# Reflected dunder ops: __radd__ etc. The inner name (after stripping __)
# starts with 'r', and the non-reflected version is the rest.
_REFLECTED_DUNDERS = {
    "radd", "rsub", "rmul", "rtruediv", "rfloordiv", "rmod", "rpow",
    "rmatmul", "rand", "ror", "rxor", "rlshift", "rrshift",
}

# DTensor handles some ops via decomposition. When the primary aten op
# (e.g. aten.matmul) has no strategy, try these alternatives.
_ATEN_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "matmul": ("mm", "bmm"),
}

# Preferred overload names, tried in order when resolving aten ops.
_PREFERRED_OVERLOADS = ("default", "Tensor")

# Cache: torch function -> (aten_op | None, reversed_args)
_aten_op_cache: dict[Callable, tuple[torch._ops.OpOverload | None, bool]] = {}


def _resolve_aten_op(
    func: Callable,
    propagator: object,
) -> tuple[torch._ops.OpOverload | None, bool]:
    """Resolve a torch function to its ATen op overload automatically.

    Uses func.__name__ to derive the aten op name, handling dunder methods
    and reflected operators. Picks the overload that DTensor's
    ShardingPropagator has a strategy for.

    Returns (aten_op, reversed_args) where reversed_args is True for
    reflected operators like __rmatmul__.
    """
    cached = _aten_op_cache.get(func)
    if cached is not None:
        return cached

    name = func.__name__
    reversed_args = False

    # Strip dunder wrappers: __add__ → add, __rmatmul__ → matmul
    if name.startswith("__") and name.endswith("__"):
        inner = name[2:-2]
        if inner in _REFLECTED_DUNDERS:
            reversed_args = True
            inner = inner[1:]
        # Detect in-place dunders: __iadd__ → add
        elif inner.startswith("i") and hasattr(torch.ops.aten, inner[1:]):
            inner = inner[1:]
        name = _DUNDER_NAME_OVERRIDES.get(inner, inner)

    packet = getattr(torch.ops.aten, name, None)
    if packet is None:
        _aten_op_cache[func] = (None, False)
        return None, False

    # Pick the overload that the propagator has a strategy for.
    for overload_name in _PREFERRED_OVERLOADS:
        op = getattr(packet, overload_name, None)
        if op is None:
            continue
        if (
            op in propagator.op_strategy_funcs
            or op in propagator.op_single_dim_strategy_funcs
            or op in propagator.op_to_rules
        ):
            _aten_op_cache[func] = (op, reversed_args)
            return op, reversed_args

    # DTensor handles some ops via decomposition (e.g. matmul → mm/bmm).
    # Try known alternative aten names.
    for alt_name in _ATEN_ALTERNATIVES.get(name, ()):
        alt_packet = getattr(torch.ops.aten, alt_name, None)
        if alt_packet is None:
            continue
        for overload_name in _PREFERRED_OVERLOADS:
            op = getattr(alt_packet, overload_name, None)
            if op is None:
                continue
            if (
                op in propagator.op_strategy_funcs
                or op in propagator.op_single_dim_strategy_funcs
                or op in propagator.op_to_rules
            ):
                _aten_op_cache[func] = (op, reversed_args)
                return op, reversed_args

    # No registered strategy found for any overload.
    _aten_op_cache[func] = (None, False)
    return None, False


class _ShardPropagator:
    """Lightweight per-axis shard propagator backed by DTensor's ShardingPropagator.

    Lazily initializes on first use: imports DTensor ops (triggering strategy
    registration) and creates a 1-D DeviceMesh for building OpSchemas.
    """

    def __init__(self):
        self._propagator = None
        # Cache 1-D meshes by axis dimension size.
        self._meshes: dict[int, object] = {}

    def _ensure_init(self):
        if self._propagator is not None:
            return
        import torch.distributed.tensor._ops  # noqa: F401  triggers registration
        from torch.distributed.tensor import DTensor

        self._propagator = DTensor._op_dispatcher.sharding_propagator

    def _get_fake_mesh(self, axis_size: int):
        """Get or create a 1-D DeviceMesh of the given size for DTensor propagation."""
        # TODO(zpcore): Need to handle multiple mesh case where multiple mesh
        # may reuse same mesh dim name.
        cached = self._meshes.get(axis_size)
        if cached is not None:
            return cached

        import torch.distributed as dist

        if not dist.is_initialized():
            import os

            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", str(axis_size))
            dist.init_process_group(backend="fake", world_size=axis_size, rank=0)

        from torch.distributed.device_mesh import DeviceMesh

        mesh = DeviceMesh("cpu", torch.arange(axis_size))
        self._meshes[axis_size] = mesh
        return mesh

    def propagate(
        self,
        func: Callable,
        axis: DeviceMeshAxis,
        axis_types: list[PerMeshAxisSpmdType],
        args: tuple,
        kwargs: dict | None = None,
    ) -> PerMeshAxisSpmdType | list[PerMeshAxisSpmdType]:
        self._ensure_init()
        aten_op, reversed_args = _resolve_aten_op(func, self._propagator)

        if aten_op is None:
            raise SpmdTypeError(
                f"No sharding strategy for {func.__name__}: "
                f"could not resolve to a registered ATen op"
            )
        try:
            return self._propagate_via_dtensor(
                aten_op, axis, args, reversed_args, kwargs=kwargs,
            )
        except SpmdTypeError:
            raise
        except Exception as e:
            raise SpmdTypeError(
                f"No valid sharding strategy for {func.__name__} "
                f"with {axis_types} on axis {format_axis(axis)}"
            ) from e

    def _propagate_via_dtensor(
        self,
        aten_op: torch._ops.OpOverload,
        axis: DeviceMeshAxis,
        args: tuple,
        reversed_args: bool = False,
        kwargs: dict | None = None,
    ) -> PerMeshAxisSpmdType | list[PerMeshAxisSpmdType]:
        """Propagate shard types through DTensor's ShardingPropagator.

        Uses pytree to walk args and replace typed tensors with DTensorSpecs,
        preserving the arg structure (important for ops like cat/stack that
        take a list of tensors).
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor._op_schema import OpSchema

        axis_size = _get_axis_dim_size(axis)
        if axis_size is None:
            raise SpmdTypeError(
                f"Cannot determine mesh dimension size for axis {format_axis(axis)}"
            )
        mesh = self._get_fake_mesh(axis_size)

        # For reversed dunder ops (__radd__ etc.), swap the first two args
        # so the OpSchema matches the ATen op's argument order.
        prop_args = args
        if reversed_args and len(args) >= 2:
            prop_args = (args[1], args[0]) + args[2:]

        # Use pytree to build args_schema: replace typed tensors with
        # DTensorSpecs, pass non-tensor args (int, float, etc.) through.
        flat_args, tree_spec = torch.utils._pytree.tree_flatten(prop_args)
        flat_schema = []
        input_axis_types = []
        for arg in flat_args:
            if isinstance(arg, torch.Tensor) and has_local_type(arg):
                typ = get_local_type(arg).get(axis, V)
                input_axis_types.append(typ)
                # DTensor expects global shapes in TensorMeta.
                shape = list(arg.shape)
                # TODO(zpcore): We need to know the global tensor shape in order
                # to build the DTensorSpec. Current code assumes even sharding.
                # Check to see if it is possible to handle uneven sharding.
                if isinstance(typ, Shard):
                    shape[typ.dim] *= axis_size
                # Recompute stride for the global shape (local stride may not
                # match after transpose or other non-contiguous views).
                global_shape = torch.Size(shape)
                stride = torch.empty(global_shape).stride()
                flat_schema.append(
                    DTensorSpec(
                        mesh=mesh,
                        placements=(_to_dt_placement(typ),),
                        tensor_meta=TensorMeta(
                            shape=global_shape,
                            stride=stride,
                            dtype=arg.dtype,
                        ),
                    )
                )
            else:
                flat_schema.append(arg)
        args_schema = torch.utils._pytree.tree_unflatten(flat_schema, tree_spec)

        # Merge non-tensor kwargs into positional args based on the aten op
        # schema.  DTensor strategy functions (e.g. sort_default_strategy)
        # expect dim/descending as positional args, but __torch_function__
        # may receive them as kwargs.
        kwargs_schema: dict = {}
        if kwargs:
            schema_args = aten_op._schema.arguments
            arg_names = [a.name for a in schema_args]
            args_list = list(args_schema)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    continue
                if k in arg_names:
                    idx = arg_names.index(k)
                    # Extend args_list with defaults if needed.
                    while len(args_list) < idx + 1:
                        args_list.append(schema_args[len(args_list)].default_value)
                    args_list[idx] = v
                else:
                    kwargs_schema[k] = v
            args_schema = tuple(args_list)

        op_schema = OpSchema(
            op=aten_op,
            args_schema=args_schema,
            kwargs_schema=kwargs_schema,
        )
        output_sharding = self._propagator.propagate_op_sharding(op_schema)

        if output_sharding.needs_redistribute:
            raise SpmdTypeError(
                f"No exact sharding strategy for {aten_op} "
                f"with {input_axis_types} on axis {format_axis(axis)}"
            )

        output_spec = output_sharding.output_spec
        if isinstance(output_spec, DTensorSpec):
            assert len(output_spec.placements) == 1, output_spec.placements
            return _from_dt_placement(output_spec.placements[0])
        if isinstance(output_spec, (tuple, list)):
            # Multi-output ops (e.g. split): one type per output tensor.
            # None entries are non-tensor outputs, kept as None.
            for spec in output_spec:
                if spec is not None:
                    assert len(spec.placements) == 1, spec.placements
            return [
                _from_dt_placement(spec.placements[0]) if spec is not None else None
                for spec in output_spec
            ]
        raise SpmdTypeError(
            f"DTensor propagator returned no output spec for {aten_op}"
        )


_shard_propagator = _ShardPropagator()


def _propagate_shard_for_axis(
    func: Callable,
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisSpmdType],
    args: tuple,
    kwargs: dict | None = None,
) -> PerMeshAxisSpmdType | list[PerMeshAxisSpmdType]:
    """Propagate S(i) through an op for one mesh axis.

    Returns a single type for single-output ops, or a list of types
    for multi-output ops (one per output tensor).

    Uses DTensor's ShardingPropagator via the ATen op mapping. Raises
    SpmdTypeError if the op cannot be resolved or has no valid strategy.
    """
    return _shard_propagator.propagate(func, axis, axis_types, args, kwargs=kwargs)


def _validate_local_global_correspondence(
    local_result: PerMeshAxisLocalSpmdType,
    global_result: PerMeshAxisGlobalSpmdType | list[PerMeshAxisGlobalSpmdType],
    axis: DeviceMeshAxis,
) -> PerMeshAxisGlobalSpmdType | list[PerMeshAxisGlobalSpmdType]:
    """Validate that local SPMD and DTensor outputs correspond.

    Expected mapping (to_local_type): S(i)→V, R→R, P→P.
    Special case: I→R (DTensor has no I; recovered to I).
    For multi-output ops, validates each output element.
    """
    if isinstance(global_result, list):
        return [
            _validate_local_global_correspondence(local_result, r, axis)
            for r in global_result
        ]
    if global_result is None:
        return None
    # I is special: DTensor has no I concept, so it produces R instead.
    if local_result is I:
        if global_result is not R:
            raise SpmdTypeError(
                f"Local SPMD produces I on axis {format_axis(axis)} but "
                f"DTensor produces {global_result}; expected R."
            )
        return I
    # For all other types, to_local_type(global) should match local.
    mapped_local = to_local_type(global_result)
    if mapped_local != local_result:
        raise SpmdTypeError(
            f"Local SPMD produces {local_result} on axis {format_axis(axis)} but "
            f"DTensor produces {global_result} (maps to {mapped_local})."
        )
    return global_result


def _infer_global_output_type(
    func: Callable,
    spec: _OpSpec | None,
    args: tuple,
    input_types_list: list[LocalSpmdType],
    shard_axes: set[DeviceMeshAxis],
    all_axes: set[DeviceMeshAxis],
    kwargs: dict | None = None,
) -> LocalSpmdType | list[LocalSpmdType]:
    """Infer output type in global SPMD when shard axes are present.

    Returns a single type dict for single-output ops, or a list of
    type dicts for multi-output ops (one per output tensor).

    For each shard axis:
    1. Replace S→V, run local SPMD to get local_result.
    2. If local SPMD rejects → propagate error.
    3. If local SPMD accepts → run DTensor, validate correspondence:
       V→S, R→R, P→P, I→R (recovered to I in output).
    Non-shard axes: delegate to existing local SPMD inference.
    """
    linearity = spec.linearity if spec is not None else OpLinearity.NONLINEAR

    # Propagate shard axes, collecting per-axis results.
    # Each result is either a single type or a list (multi-output).
    shard_results: dict[DeviceMeshAxis, PerMeshAxisSpmdType | list[PerMeshAxisSpmdType]] = {}
    # Track number of outputs for multi-output ops.
    n_outputs = None
    for axis in shard_axes:
        axis_types = [typ.get(axis, V) for typ in input_types_list]

        # Step 1: Replace S→V, check compatibility via local SPMD.
        local_axis_types = [to_local_type(t) for t in axis_types]
        local_result = infer_local_type_for_axis(
            axis, local_axis_types, linearity=linearity
        )
        # Local SPMD accepted. Use DTensor for the global output.
        result = _propagate_shard_for_axis(func, axis, axis_types, args, kwargs=kwargs)
        # Validate local↔global correspondence: V→S, R→R, P→P, I→R.
        # If local SPMD produces I, DTensor gives R (it has no I);
        # recover I in the final output.
        result = _validate_local_global_correspondence(
            local_result, result, axis
        )
        shard_results[axis] = result
        if isinstance(result, list):
            if n_outputs is not None and len(result) != n_outputs:
                raise ValueError(
                    f"Op {func.__name__} has inconsistent number of outputs "
                    f"on axis {format_axis(axis)}: {n_outputs} vs {len(result)}"
                )
            else:
                n_outputs = len(result)

    # Non-shard axes: single type shared across all outputs.
    non_shard_output: LocalSpmdType = {}
    non_shard_axes = all_axes - shard_axes
    if non_shard_axes:
        local_types = [
            {a: t for a, t in typ.items() if a not in shard_axes}
            for typ in input_types_list
        ]
        # infer_output_type derives a single type regardless of output count.
        # When local SPMD produces V, global SPMD maps V → S(i) per output.
        non_shard_output = infer_output_type(local_types, linearity=linearity)

    # Case: multi-output op.
    if n_outputs is not None:
        # Multi-output: build per-output type dicts.
        output_types: list[LocalSpmdType] = [
            dict(non_shard_output) for _ in range(n_outputs)
        ]
        for axis, result in shard_results.items():
            if isinstance(result, list):
                for i, rt in enumerate(result):
                    output_types[i][axis] = rt
            else:
                for ot in output_types:
                    ot[axis] = result
        return output_types

    # Case: Single output op.
    output_type: LocalSpmdType = dict(non_shard_output)
    for axis, result in shard_results.items():
        output_type[axis] = result
    return output_type


# =============================================================================
# TorchFunctionMode for SPMD Type Tracking
# =============================================================================

# Functions that are autograd/metadata bookkeeping, not tensor math.
# These go through __torch_function__ but should not trigger type inference.
_AUTOGRAD_PASSTHROUGH = {
    torch.Tensor.backward,
    torch.Tensor.requires_grad_,
    torch.Tensor.retain_grad,
}


class SpmdTypeMode(torch.overrides.TorchFunctionMode):
    """
    TorchFunctionMode for tracking SPMD types on tensors.

    When active, this mode intercepts torch operations and propagates
    SPMD types from inputs to outputs according to the typing rules.

    For SPMD collectives and local ops (all_reduce, all_gather, etc.), the
    mode validates that the input tensor's type on the relevant mesh axis
    matches the declared ``src`` and sets the output type to ``dst``.  Types
    on all other mesh axes are copied through unchanged.

    Type checking runs *after* the function executes so that runtime errors
    (shape mismatches, invalid arguments) surface before type errors.

    Args:
        strict: If True, raises ``SpmdTypeError`` when a regular torch op
            mixes typed and untyped tensor operands.  Once all tensors are
            annotated, keep this on to prevent unannotated tensors from
            silently slipping through.
    """

    def __init__(
        self,
        strict: bool = True,
        global_mode: bool = False,
        mesh_dim_sizes: dict[str, int] | None = None,
    ):
        super().__init__()
        self.strict = strict
        self.global_mode = global_mode
        self.mesh_dim_sizes = mesh_dim_sizes
        self._prev_global_mode = False
        self._prev_mesh_dim_sizes: dict[str, int] | None = None

    def __enter__(self):
        self._prev_global_mode = _is_global_mode()
        _set_global_mode(self.global_mode)
        if self.global_mode and self.mesh_dim_sizes is not None:
            self._prev_mesh_dim_sizes = _get_mesh_dim_sizes()
            _set_mesh_dim_sizes(self.mesh_dim_sizes)
        return super().__enter__()

    def __exit__(self, *args):
        _set_global_mode(self._prev_global_mode)
        if self.global_mode and self.mesh_dim_sizes is not None:
            _set_mesh_dim_sizes(self._prev_mesh_dim_sizes)
        return super().__exit__(*args)

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}

        # Autograd bookkeeping -- not tensor math, skip type inference.
        if func in _AUTOGRAD_PASSTHROUGH:
            return func(*args, **kwargs)

        # Property access (e.g. .grad, .data, .shape) -- not tensor math.
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # In global mode, only redistribute is allowed. Low-level collectives
        # (all_reduce, all_gather, etc.) are implementation details that
        # redistribute delegates to.
        if _is_global_mode() and func in _SPMD_FUNCTION_DEFAULTS:
            if func is not redistribute:
                raise SpmdTypeError(
                    f"{func.__name__} is not allowed in global SPMD mode. "
                    f"Use redistribute() instead."
                )
            x = args[0]
            if has_local_type(x):
                if self.strict:
                    _check_all_typed(args, kwargs)
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = args[1] if len(args) > 1 else kwargs["axis"]
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                input_type = get_local_type(x).get(axis, V)
                if src is not None and input_type != src:
                    raise SpmdTypeError(
                        f"{func.__name__}: expected input type {src} on axis "
                        f"{format_axis(axis)}, got {input_type}"
                    )
                output_type = get_local_type(x).copy()
                if dst is not None:
                    output_type[axis] = dst
                result = func(*args, **kwargs)
                setattr(result, _SPMD_TYPE_ATTR, output_type)
                return result

        # Run the function first (catches runtime errors before type errors).
        # This means collectives execute before type checking, but that's fine:
        # type checking typically runs with a fake PG where comms are free.
        # In global mode, shape mismatches from type-level-only collectives
        # are expected; fall back to cloning the first typed input.
        if _is_global_mode():
            try:
                result = func(*args, **kwargs)
            except RuntimeError as e:
                first_input = next(_iter_tensor_args(args, kwargs), None)
                if first_input is None:
                    raise
                log.debug(
                    "Global SPMD: %s raised %s, cloning first input for "
                    "type-level propagation",
                    getattr(func, "__name__", func), e,
                )
                result = first_input.clone()
        else:
            result = func(*args, **kwargs)

        # Strict mode: all tensor operands must be annotated (applies to both
        # SPMD collectives and regular torch ops).
        if self.strict:
            _check_all_typed(args, kwargs)

        if func in _SPMD_FUNCTION_DEFAULTS:
            # Special spmd collective/reinterpret/collect
            x = args[0]
            if has_local_type(x):
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = args[1] if len(args) > 1 else kwargs["axis"]
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                # In local mode, decay Shard to Varying for type checking.
                if isinstance(src, Shard):
                    src = V
                if isinstance(dst, Shard):
                    dst = V

                # Validate input type on the axis matches src
                input_type = get_axis_local_type(x, axis)
                if src is not None:
                    if input_type != src:
                        raise SpmdTypeError(
                            f"{func.__name__}: expected input type {src} on axis "
                            f"{format_axis(axis)}, got {input_type}"
                        )

                # Build output types: copy all axes from input, override this axis
                output_type = get_local_type(x).copy()
                if dst is not None:
                    output_type[axis] = dst
                _set_local_type(result, output_type)
        else:
            # Regular torch op: propagate types according to non-comms rules
            spec = _OP_REGISTRY.get(func)
            input_types_list = _collect_input_types(args, kwargs, spec)
            if input_types_list:
                # Global mode: check for shard axes requiring propagation.
                if _is_global_mode():
                    all_axes: set[DeviceMeshAxis] = set()
                    for typ in input_types_list:
                        all_axes.update(typ.keys())
                    shard_axes = {
                        axis
                        for axis in all_axes
                        if any(
                            isinstance(typ.get(axis, V), Shard)
                            for typ in input_types_list
                        )
                    }
                    if shard_axes:
                        output_type = _infer_global_output_type(
                            func, spec, args, input_types_list,
                            shard_axes, all_axes, kwargs=kwargs,
                        )
                        if isinstance(output_type, list):
                            _set_result_types(result, output_type)
                        else:
                            _set_result_type(result, output_type)
                        return result

                # Local SPMD path (also used as fallback when no shard axes).
                decomp_rule = _DECOMP_TYPE_RULES.get(func)
                if decomp_rule is not None:
                    output_type = decomp_rule(*input_types_list)
                else:
                    linearity = (
                        spec.linearity if spec is not None else OpLinearity.NONLINEAR
                    )
                    if spec is not None and spec.fixed_args:
                        input_types_list = _apply_fixed_args(
                            func, args, kwargs, spec, input_types_list
                        )
                    output_type = infer_output_type(
                        input_types_list, linearity=linearity
                    )
                _set_result_type(result, output_type)

        return result


# =============================================================================
# Local/Global SPMD Transition
# =============================================================================


def local_map(
    in_type: list[dict[DeviceMeshAxis, PerMeshAxisSpmdType]],
    out_type: dict[DeviceMeshAxis, PerMeshAxisSpmdType],
):
    """Decorator factory for local SPMD regions within global SPMD.

    On entry: validates inputs match in_type, decays S(i) -> V.
    Inside: runs in local SPMD mode (only R/I/V/P propagate).
    On exit: re-annotates outputs with S(i) from out_type, restores inputs.

    Args:
        in_type: Expected per-axis types for each positional tensor arg.
            Length must match the number of positional args at call time.
        out_type: Per-axis types to stamp on each output tensor.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if len(args) != len(in_type):
                raise SpmdTypeError(
                    f"local_map: expected {len(in_type)} positional args, "
                    f"got {len(args)}"
                )

            # Validate inputs match in_type
            for arg, expected_type in zip(args, in_type):
                stored = get_local_type(arg)
                for axis, typ in expected_type.items():
                    actual = stored.get(axis, V)
                    if actual != typ:
                        raise SpmdTypeError(
                            f"local_map: input has {actual} on axis "
                            f"{format_axis(axis)}, expected {typ}"
                        )

            # Save input types, then decay S -> V for local mode.
            saved_types = []
            for arg, expected_type in zip(args, in_type):
                saved_types.append(get_local_type(arg).copy())
                new_type = get_local_type(arg).copy()
                for axis, typ in expected_type.items():
                    local = to_local_type(typ)
                    if local is V:
                        new_type.pop(axis, None)
                    else:
                        new_type[axis] = local
                _set_local_type(arg, new_type)

            # Temporarily disable global mode
            prev = _is_global_mode()
            _set_global_mode(False)
            try:
                result = fn(*args, **kwargs)
            finally:
                _set_global_mode(prev)

            # Re-annotate outputs with out_type (recursive, following
            # DTensor's OpDispatcher.wrap pattern).
            def apply_out_type(res):
                if isinstance(res, torch.Tensor):
                    current = get_local_type(res) if has_local_type(res) else {}
                    for axis, typ in out_type.items():
                        current[axis] = typ
                    _set_local_type(res, current)
                elif isinstance(res, (list, tuple)):
                    for item in res:
                        apply_out_type(item)

            apply_out_type(result)

            # Restore input types (bypass validation since these are known-good
            # saved types that may contain Shard values from global mode).
            for arg, saved in zip(args, saved_types):
                setattr(arg, _SPMD_TYPE_ATTR, saved)

            return result

        return wrapper

    return decorator
