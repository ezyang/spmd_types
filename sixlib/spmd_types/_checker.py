"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Literal, NamedTuple, Optional, overload

import torch
import torch.overrides
from sixlib.spmd_types import _dist
from sixlib.spmd_types._collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
)
from sixlib.spmd_types._frame import _get_user_frame
from sixlib.spmd_types._local import convert, invariant_to_replicate, reinterpret
from sixlib.spmd_types._scalar import _unwrap_args, Scalar
from sixlib.spmd_types._traceback import _filter_and_reraise, api_boundary
from sixlib.spmd_types._type_attr import (
    _LOCAL_TYPE_ATTR,
    get_axis_local_type,  # noqa: F401 (re-exported for tests)
    get_local_type,
    set_local_type as _set_local_type_raw,
)
from sixlib.spmd_types.types import (
    _canonicalize_shard,
    DeviceMeshAxis,
    format_axis,
    I,
    LocalSpmdType,
    normalize_axis,
    normalize_local_type,
    P,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    R,
    Shard,
    shard_types_to_partition_spec,
    SpmdTypeError,
    V,
)
from torch.distributed.tensor import _api as _dt_api, DTensor, Placement, Replicate
from torch.distributed.tensor._api import (
    _FromTorchTensor,
    _ToTorchTensor,
)

if hasattr(_dt_api, "_normalize_placements_for_grad"):
    _normalize_placements_for_grad = _dt_api._normalize_placements_for_grad
else:

    def _normalize_placements_for_grad(
        placements: tuple[Placement, ...],
    ) -> tuple[Placement, ...]:
        normalized: list[Placement] = []
        for p in placements:
            if p.is_partial():
                normalized.append(Replicate())
            else:
                normalized.append(p)
        return tuple(normalized)


from torch.overrides import handle_torch_function, has_torch_function


def has_local_type(tensor: torch.Tensor) -> bool:
    """Return True if the tensor has an SPMD type annotation.

    Distinguishes between truly untyped tensors (no attribute) and factory
    tensors (attribute set to ``{}``).
    """
    return hasattr(tensor, _LOCAL_TYPE_ATTR)


# =============================================================================
# Trace mode: set SPMD_TYPES_TRACE=1 to log every non-trivial operator.
# =============================================================================

_trace_logger = logging.getLogger(__name__ + ".trace")
_TRACE = os.environ.get("SPMD_TYPES_TRACE", "") == "1"


@contextmanager
def trace(enabled: bool = True):
    """Context manager to enable or disable SPMD type trace logging.

    When enabled, every non-trivial tensor operator logs its name, input
    SPMD types, output SPMD type, and the user-code callsite to the
    ``sixlib.spmd_types._checker.trace`` logger at INFO level.

    Example::

        import logging
        logging.basicConfig()
        with typecheck(), trace():
            z = x + y  # logs: my_file.py:42  add({dp: R}, {dp: V}) -> {dp: V}

    Can also be used to temporarily suppress tracing when the envvar
    ``SPMD_TYPES_TRACE=1`` is set::

        with trace(enabled=False):
            ...  # no trace output
    """
    global _TRACE
    old = _TRACE
    _TRACE = enabled
    try:
        yield
    finally:
        _TRACE = old


def _format_type(t: object) -> str:
    """Format a per-mesh-axis type dict for trace output."""
    if isinstance(t, dict):
        if not t:
            return "{}"
        parts = []
        for k, v in t.items():
            # Use the short name (R, I, V, P) instead of the full enum repr.
            v_str = v.name if isinstance(v, PerMeshAxisLocalSpmdType) else repr(v)
            parts.append(f"{k}: {v_str}")
        return "{" + ", ".join(parts) + "}"
    return repr(t)


def _trace_op(
    func: object,
    input_types: list[LocalSpmdType],
    output_type: LocalSpmdType | None,
) -> None:
    """Log a trace line for a non-trivial tensor operator."""
    name = getattr(func, "__name__", None) or getattr(func, "__qualname__", repr(func))
    inputs_str = ", ".join(_format_type(t) for t in input_types)
    out_str = _format_type(output_type)
    loc = _get_user_frame()
    _trace_logger.info("%s  %s(%s) -> %s", loc, name, inputs_str, out_str)


# =============================================================================
# Scalar Sentinel
# =============================================================================


class _ScalarType:
    """Sentinel for Python scalars in SPMD type inference.

    We assume a Python scalar is the same value on all ranks and carries no
    gradient.  This assumption can be wrong: a rank-dependent scalar (e.g.
    ``rank / world_size``) is really Varying, and a scalar that is the sum of
    per-rank contributions is really Partial.  We choose to assume Replicate
    anyway because (1) the vast majority of scalars in practice *are* the same
    on every rank, and (2) requiring users to annotate every literal ``2.0``
    or ``eps`` would be extremely noisy for little safety gain.  A future
    improvement could add an explicit wrapper (e.g. ``Varying(scalar)``) for
    the rare rank-dependent case; until then Dynamo tracing with SymInt offers
    partial safety.

    Given the assumption, _Scalar is compatible with both R and I -- analogous
    to how Python scalars are "weak" in dtype promotion and do not determine
    the specific dtype.

    _Scalar participates in linearity validation (P + scalar is affine, not
    linear) but does not influence the inferred output type (R vs I).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Scalar"


_Scalar = _ScalarType()


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
        "convert(tensor, {axis_arg}, src=I, dst=R)",
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
    axis_arg = format_axis(axis)

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
    lines = [
        base_msg,
        "Are you missing a collective or a reinterpret/convert call? e.g.,",
    ]
    for operation, consequence, from_type in suggestions:
        lines.append(
            f"  {operation} on the {_TYPE_FULL_NAMES[from_type]} operand ({consequence})"
        )
    return "\n".join(lines)


# =============================================================================
# SPMD Type Tracking on Tensors
# =============================================================================


def _validate(type: LocalSpmdType) -> LocalSpmdType:
    """Validate a LocalSpmdType and normalize axis keys.

    Extends ``normalize_local_type`` with additional checks for internal
    sentinel types (_Scalar, Shard) that must not be stored on tensors.

    Args:
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P),
            or is an internal sentinel type.
    """
    # Pre-check for internal sentinel types that normalize_local_type
    # does not know about (they live in _checker.py, not types.py).
    for axis, typ in type.items():
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            if typ is _Scalar:
                raise TypeError(
                    f"_Scalar sentinel on axis {format_axis(axis)} must not be stored "
                    f"on a tensor. _Scalar is internal to type inference; it should "
                    f"be filtered out by infer_output_type before reaching a tensor."
                )
            if isinstance(typ, Shard):
                raise TypeError(
                    f"Shard type {typ!r} on axis {format_axis(axis)} cannot be stored "
                    f"as a local SPMD type. Shard is only valid as src/dst in "
                    f"collective operations. Use V instead for local type tracking."
                )
            # Fall through to normalize_local_type for the generic error.
    return normalize_local_type(type)


# get_local_type, get_axis_local_type are imported from _type_attr (a leaf
# module shared with _raw_dist to avoid a Buck cycle).


def _set_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Set SPMD type on a tensor (internal). Validates and returns tensor."""
    return _set_local_type_raw(tensor, _validate(type))


# Attribute name for storing the PartitionSpec on tensors (global SPMD only).
# This is the single source of truth for which mesh axes shard which tensor dims.
_PARTITION_SPEC_ATTR = "_partition_spec"


def get_partition_spec(tensor: torch.Tensor) -> PartitionSpec | None:
    """Get the PartitionSpec stored on a tensor (global SPMD only).

    Returns None if no global shard info is present. The PartitionSpec
    is the single source of truth for which mesh axes shard which tensor
    dims in global SPMD.

    Args:
        tensor: The tensor to retrieve the PartitionSpec from.
    """
    return getattr(tensor, _PARTITION_SPEC_ATTR, None)


def _set_partition_spec(tensor: torch.Tensor, spec: PartitionSpec | None) -> None:
    """Set or clear the partition spec on a single tensor."""
    if spec is not None:
        setattr(tensor, _PARTITION_SPEC_ATTR, spec)
    elif hasattr(tensor, _PARTITION_SPEC_ATTR):
        delattr(tensor, _PARTITION_SPEC_ATTR)


@overload
def assert_type(
    tensor: torch.Tensor,
    type: LocalSpmdType,
    partition_spec: PartitionSpec | None = ...,
) -> torch.Tensor: ...


@overload
def assert_type(
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes,
) -> torch.Tensor: ...


@api_boundary
def assert_type(  # noqa: C901
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes,
    partition_spec: PartitionSpec | None = None,
) -> torch.Tensor:
    """Assert or set the SPMD type on a tensor.

    If the tensor has no SPMD type, sets it. If the tensor already has an SPMD
    type, checks compatibility using refinement semantics (see below).

    Two calling conventions (see overloads):

    1. ``assert_type(tensor, {axis: R/I/V/P, ...}, partition_spec=...)``
       Explicit local types with optional PartitionSpec for shard metadata.

    2. ``assert_type(tensor, {axis: S(i), ...})`` S(i) entries are automatically
       converted to V + PartitionSpec. Cannot be combined with an explicit
       ``partition_spec``.

    S(i) always stores a PartitionSpec regardless of whether the axis is in
    global SPMD mode. Global axes only affect whether S(i) propagates through
    ops (via DTensor); storage is unconditional.

    Refinement semantics (re-check on already-typed tensors):

    When called on a tensor that already has SPMD types, ``assert_type`` checks
    consistency rather than overwriting. For local types (R/I/P), the existing
    and new values must match exactly. For shard metadata, S(i) is a refinement
    of V -- it adds information about which tensor dimension is sharded without
    changing the local type (both are V locally). A re-check may add new shard
    info but must not contradict existing info.

    Worked examples:

    - V then S(i): OK, stores the shard info (refinement).
    - S(i) then V: OK, keeps existing shard info (V is less specific).
    - S(i) then S(i): OK (consistent).
    - S(i) then S(j) on same axis: SpmdTypeError (contradicts).
    - {tp: S(0)} then {dp: S(1)}: OK, merges to PartitionSpec(tp, dp).
    - {tp: S(0)} then {dp: S(0)}: SpmdTypeError (multi-axis same dim requires
      explicit PartitionSpec, e.g. PartitionSpec((tp, dp), None)).
    - {dp: S(0)} then PartitionSpec(dp, None): OK (consistent).
    - {dp: S(0)} then PartitionSpec(dp, tp): OK, adds new shard info.
    - {dp: S(0)} then PartitionSpec((dp, tp), None): SpmdTypeError, touching
      existing S(0) ordering info.
    - PartitionSpec(dp, None) then PartitionSpec(dp, tp): OK, adds new
      shard info for axis tp.
    - PartitionSpec(dp, None) then PartitionSpec((dp, tp), None): SpmdTypeError,
      touching existing S(0) ordering info.
    - PartitionSpec((dp, tp), None) then {dp: S(0)}: SpmdTypeError,
      unresolved order on S(0).
    - PartitionSpec ordering matters: (tp, dp) != (dp, tp).

    Args:
        tensor: The tensor to assert or set SPMD type on. type: A dict mapping
        mesh axes to per-axis SPMD types.
            Accepts R, I, V, P, or S(i). S(i) entries are syntax sugar for
            setting V on the axis and storing a PartitionSpec that maps tensor
            dim ``i`` to that mesh axis.
        partition_spec: Optional PartitionSpec describing how tensor
            dimensions map to mesh axes for Varying dimensions. Mutually
            exclusive with S(i) entries in ``type``.

    Raises:
        SpmdTypeError: If S(i) and partition_spec are both provided,
            if partition_spec length doesn't match tensor ndim, if a type dict
            axis conflicts with partition_spec, or if a re-check has conflicting
            PartitionSpec info.
        SpmdTypeError: If existing local SPMD type doesn't match.
    """
    ############ Validate partition_spec length ############
    if partition_spec is not None:
        if len(partition_spec) != tensor.ndim:
            raise SpmdTypeError(
                f"PartitionSpec length {len(partition_spec)} doesn't match "
                f"tensor ndim {tensor.ndim}"
            )

    ############ Build canonical LocalSpmdType + auto PartitionSpec ############
    # Single pass: normalize S(i) dims, separate into local types vs shards,
    # and check for multi-axis-same-dim conflicts.
    local_type: LocalSpmdType = {}
    axis_to_dims: dict[DeviceMeshAxis, Shard] = {}
    dim_to_axes: dict[int, list[DeviceMeshAxis]] = {}
    for axis, typ in type.items():
        axis = normalize_axis(axis)
        typ = _canonicalize_shard(typ, tensor.ndim)
        if isinstance(typ, Shard):
            dim_to_axes.setdefault(typ.dim, []).append(axis)
            axis_to_dims[axis] = typ
            local_type[axis] = V
        else:
            local_type[axis] = typ

    # Enforce overload contract: S(i) and partition_spec are mutually exclusive.
    if axis_to_dims and partition_spec is not None:
        raise SpmdTypeError(
            "Cannot use S(i) in type dict and partition_spec at the same time. "
            "Use either S(i) in the type dict or an explicit PartitionSpec."
        )

    # Convert S(i) entries to PartitionSpec.
    if axis_to_dims:
        # Forbid sharding the same tensor dim on multiple mesh axes without
        # an explicit PartitionSpec (which specifies the axis ordering).
        for dim, axes in dim_to_axes.items():
            if len(axes) > 1:
                names = ", ".join(format_axis(a) for a in axes)
                raise SpmdTypeError(
                    f"Tensor dim {dim} is sharded on multiple axes "
                    f"({names}). Use an explicit PartitionSpec "
                    f"to specify the axis ordering."
                )
        partition_spec = shard_types_to_partition_spec(axis_to_dims, tensor.ndim)

    ############ Fill V for axes in partition_spec ############
    if partition_spec is not None:
        for entry in partition_spec:
            if entry is None:
                continue
            axes = entry if isinstance(entry, tuple) else (entry,)
            for axis in axes:
                axis = normalize_axis(axis)
                axis_type = local_type.get(axis)
                if axis_type is not None and axis_type is not V:
                    raise SpmdTypeError(
                        f"Mesh axis {format_axis(axis)} appears in "
                        f"partition_spec (implying Varying/Shard) but is "
                        f"specified as {axis_type} in type."
                    )
                local_type[axis] = V

    _validate(local_type)

    ############ Set or check ############
    if not has_local_type(tensor):
        _set_local_type(tensor, local_type)
        _set_partition_spec(tensor, partition_spec)
        if _TRACE:
            _trace_op(assert_type, [{}], local_type)
        return tensor

    # 1. Re-check: compare local types.
    # Only axes present in canonical are checked; extra axes in existing are
    # ignored (partial re-check). New axes are merged in. Asserting V on an
    # axis that already has a type (R/I/P) is allowed (V is less specific).
    existing_local = get_local_type(tensor)  # existing_local is mutable.
    old = dict(existing_local) if _TRACE else None
    for axis, typ in local_type.items():
        if axis not in existing_local:
            existing_local[axis] = typ  # New axis: merge in place.
            continue
        existing_typ = existing_local[axis]
        if existing_typ != typ:
            raise SpmdTypeError(
                f"SPMD type mismatch on axis {format_axis(axis)}: "
                f"tensor has {existing_typ}, expected {typ}"
            )
    # 2. Re-check PartitionSpec: refinement semantics.
    # Only None -> sharded refinement is allowed; adding a new mesh axis to an
    # already-sharded dim requires an explicit PartitionSpec upfront.
    existing_spec = getattr(tensor, _PARTITION_SPEC_ATTR, None)
    if partition_spec is not None:
        if existing_spec is not None:
            # Build map from mesh axis to tensor dim for conflict detection.
            existing_axis_to_dim: dict[DeviceMeshAxis, int] = {}
            for dim, entry in enumerate(existing_spec):
                if entry is not None:
                    axes = entry if isinstance(entry, tuple) else (entry,)
                    for a in axes:
                        existing_axis_to_dim[a] = dim

            merged_entries = list(existing_spec)
            for dim, (ex_spec, new_spec) in enumerate(
                zip(existing_spec, partition_spec)
            ):
                if new_spec is None:
                    continue  # New spec doesn't constrain this dim.
                if ex_spec is None:
                    # Check new axis isn't already at another dim.
                    new_axes = new_spec if isinstance(new_spec, tuple) else (new_spec,)
                    for a in new_axes:
                        if a in existing_axis_to_dim:
                            raise SpmdTypeError(
                                f"PartitionSpec conflict: axis "
                                f"{format_axis(a)} already shards dim "
                                f"{existing_axis_to_dim[a]},"
                                f" cannot also shard dim {dim}"
                            )
                    merged_entries[dim] = new_spec
                    continue
                if ex_spec != new_spec:
                    raise SpmdTypeError(
                        f"PartitionSpec conflict at dim {dim}: "
                        f"tensor has {ex_spec!r}, new assert has {new_spec!r}"
                    )
            setattr(tensor, _PARTITION_SPEC_ATTR, PartitionSpec(*merged_entries))
        else:
            # Refinement: V -> S(i). Store the new spec.
            setattr(tensor, _PARTITION_SPEC_ATTR, partition_spec)
    if _TRACE:
        _trace_op(assert_type, [old], existing_local)
    return tensor


def assert_local_type(tensor: torch.Tensor, type: PerMeshAxisSpmdTypes) -> torch.Tensor:
    """Deprecated: use ``assert_type`` instead."""
    return assert_type(tensor, type)


@api_boundary
def mutate_type(
    tensor: torch.Tensor,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
) -> torch.Tensor:
    """Change the SPMD type of a single mesh axis on an already-annotated tensor.

    Unlike ``assert_type``, this function *overwrites* the existing type on
    ``axis``.  The caller must specify the expected current type (``src``) to
    prevent silent corruption; a ``SpmdTypeError`` is raised if the tensor's
    current type on ``axis`` does not match ``src``.

    This is intended for internal use in low-level parallelism primitives
    where a buffer legitimately changes its distribution semantics in place
    (e.g. an all-gather output buffer that transitions from S(0) to R).

    With D95084345 (raw dist type rules), most collectives can be type-checked
    automatically. mutate_type is still needed for updating spmd types on
    *views* into a buffer after an in-place collective has changed the
    buffer's semantics.

    Args:
        tensor: The tensor whose type to mutate.  Must already have a local
            type annotation.
        axis: The mesh axis (MeshAxis or ProcessGroup) to modify.
        src: The expected current type on ``axis``.  Raises if it does not
            match.  Accepts R, I, V, P, or S(i) (S(i) is compared as V).
        dst: The new type to set on ``axis``.  Accepts R, I, V, P, or S(i)
            (S(i) is stored as V).

    Returns:
        The tensor for chaining.

    Raises:
        SpmdTypeError: If the tensor has no type, the axis is missing,
            or the current type does not match ``src``.
    """
    axis = normalize_axis(axis)
    if axis.size() == 1:
        return tensor  # singleton axes are not tracked
    local_type = get_local_type(tensor)
    if axis not in local_type:
        raise SpmdTypeError(
            f"mutate_type: axis {format_axis(axis)} not found in tensor's "
            f"SPMD type. Tensor has axes: {set(local_type.keys())}"
        )

    # Normalize S(i) -> V for comparison and storage
    src_local = V if isinstance(src, Shard) else src
    dst_local = V if isinstance(dst, Shard) else dst

    current = local_type[axis]
    if current is not src_local:
        raise SpmdTypeError(
            f"mutate_type: expected current type {src_local} on axis "
            f"{format_axis(axis)}, got {current}"
        )

    new_type = dict(local_type)
    new_type[axis] = dst_local
    return _set_local_type(tensor, new_type)


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

    type_set = set(axis_types)

    # Check type compatibility and infer output type.
    #
    # _Scalar is compatible with R, I, and V (adopts the tensor type).
    # With P it depends on linearity: P * scalar is valid (MULTILINEAR,
    # scaling preserves partial-sum), but P + scalar is affine (LINEAR,
    # the constant gets summed N times across ranks).
    #
    #   {R}, {R, _Scalar} -> R           {I}, {I, _Scalar} -> I
    #   {V}, {V, _Scalar}, {R, V, ...} -> V
    #   {P} -> P (linearity checks)
    #   {P, _Scalar} -> MULTILINEAR: P, LINEAR: error (affine)
    #   {P, R, ...} -> MULTILINEAR single-P: P, else: error

    if type_set <= {R, _Scalar}:
        inferred_type = R
    elif type_set <= {I, _Scalar}:
        inferred_type = I
    elif type_set <= {R, V, _Scalar}:
        inferred_type = V
    elif I in type_set:
        raise SpmdTypeError(
            f"Invariant type on axis {format_axis(axis)} cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif P in type_set:
        # All P-related inference is handled here.
        non_p = type_set - {P, _Scalar}  # real non-P tensor types
        has_scalar = _Scalar in type_set
        p_count = sum(1 for t in axis_types if t is P)

        # P + V is always invalid regardless of linearity.
        if non_p - {R}:
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} cannot combine with "
                f"Varying. Reduce Partial first (all_reduce -> R, or "
                f"reduce_scatter -> V). Found types: {axis_types}"
            )

        assert type_set <= {P, R, _Scalar}, type_set
        if linearity is OpLinearity.NONLINEAR:
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} cannot propagate "
                f"through non-linear ops (non-linear of a partial sum != "
                f"partial sum of non-linear). Reduce first with all_reduce "
                f"or reduce_scatter. Found types: {axis_types}"
            )
        if linearity is OpLinearity.LINEAR and (non_p or has_scalar):
            # P + R is invalid: R contributes the same value N times.
            # P + scalar is affine: the constant gets summed N times.
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} in a linear op "
                f"requires all operands to be Partial (sum of partial sums "
                f"is a partial sum, but adding a Replicate or scalar value "
                f"makes the result affine -- the non-partial term gets "
                f"summed N times across ranks). "
                f"Found types: {axis_types}"
            )
        if linearity is OpLinearity.MULTILINEAR and p_count > 1:
            raise SpmdTypeError(
                f"Partial in multiple factors of multilinear op on axis "
                f"{format_axis(axis)} is forbidden. Reduce all but one "
                f"factor first. Found types: {axis_types}"
            )
        # Valid cases that reach here:
        # - LINEAR, all-P: sum of partial sums is still partial
        # - MULTILINEAR, single P with R|scalar: scaling preserves partial
        inferred_type = P
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
        axis: The mesh axis (for error messages)
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

    In strict mode, raises ``SpmdTypeError`` when an operand is missing an
    axis.  In permissive mode, operands missing an axis are skipped and the
    output type for that axis is inferred from the remaining operands.

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

    strict = _state.is_strict()

    # Infer output type for each axis
    output_type: LocalSpmdType = {}
    for axis in all_axes:
        axis_types = []
        for typ in input_types_list:
            if axis not in typ:
                if strict:
                    raise SpmdTypeError(
                        f"Operand missing axis {format_axis(axis)}. "
                        f"Operand has axes {set(typ.keys())}, "
                        f"but the union of all operand axes is {all_axes}. "
                        f"All operands must be annotated on the same set of "
                        f"mesh axes."
                    )
                continue  # permissive: skip this operand for this axis
            axis_types.append(typ[axis])

        if not axis_types:
            continue  # all operands missing this axis

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


_SCALAR_TYPES = (int, float, complex, bool, str, type(None))
_LEAF_TYPES = (torch.Tensor, *_SCALAR_TYPES)


def _iter_tensors_in(val: object):
    """Yield tensors from *val*, fast-pathing common leaf/flat cases.

    Direct tensors and flat list/tuple of tensors are handled without pytree.
    Anything else (nested containers, custom pytree-registered types) falls
    back to ``torch.utils._pytree.tree_flatten``.
    """
    if isinstance(val, torch.Tensor):
        yield val
        return
    if isinstance(val, (list, tuple)):
        if all(isinstance(item, _LEAF_TYPES) for item in val):
            # Fast path: flat list/tuple of tensors and scalar leaves.
            for item in val:
                if isinstance(item, torch.Tensor):
                    yield item
            return
        # Unknown or nested types present -- flatten via pytree to handle
        # nested containers and custom pytree-registered types.
        flat, _ = torch.utils._pytree.tree_flatten(val)
        for item in flat:
            if isinstance(item, torch.Tensor):
                yield item
        return
    if isinstance(val, _SCALAR_TYPES):
        return
    # Unknown type -- fall back to pytree.
    flat, _ = torch.utils._pytree.tree_flatten(val)
    for item in flat:
        if isinstance(item, torch.Tensor):
            yield item


class _RawArgEntry(NamedTuple):
    """One argument (positional or keyword) captured for error display."""

    location: str | None
    value: object


class _ArgInfo(NamedTuple):
    """Classification and collected types from all tensor arguments.

    Produced by ``_classify_args`` in a single pass over args/kwargs.

    Attributes:
        tensor_types: Types from all tensors (in iteration order).
            Untyped tensors contribute ``{}`` (unknown on all axes).
        raw_entries: All arguments (tensor and non-tensor), in call order,
            captured so error formatting can be deferred until a type error
            actually occurs.
    """

    tensor_types: list[LocalSpmdType]
    raw_entries: list[_RawArgEntry]


def _format_tensor_for_context(t: torch.Tensor) -> str:
    """Format a single tensor for error context display."""
    from torch.utils._dtype_abbrs import dtype_abbrs

    lt = get_local_type(t)
    dtype_str = dtype_abbrs.get(t.dtype, str(t.dtype).removeprefix("torch."))
    shape_str = ", ".join(str(d) for d in t.shape)
    type_items = ", ".join(f"{format_axis(axis)}: {typ!r}" for axis, typ in lt.items())
    return f"{dtype_str}[{shape_str}] {{{type_items}}}"


def _format_non_tensor_for_context(val: object) -> str:
    """Format a non-tensor argument for error context display.

    Special-cases ProcessGroup to show the mesh axis name instead of the
    raw repr.
    """
    if isinstance(val, _dist.dist.ProcessGroup):
        return repr(normalize_axis(val))
    return repr(val)


def _format_arg_for_context(val: object) -> str:
    """Format a single argument value for error context display.

    Handles tensors, lists of tensors, ProcessGroups, and other values.
    """
    if isinstance(val, torch.Tensor):
        return _format_tensor_for_context(val)
    if isinstance(val, (list, tuple)):
        has_tensors = any(isinstance(item, torch.Tensor) for item in val)
        if has_tensors:
            bracket = "[" if isinstance(val, list) else "("
            close = "]" if isinstance(val, list) else ")"
            items = []
            for item in val:
                if isinstance(item, torch.Tensor):
                    items.append(_format_tensor_for_context(item))
                else:
                    items.append(_format_non_tensor_for_context(item))
            # If all items are identical, abbreviate: [item] * N
            if len(items) > 1 and all(it == items[0] for it in items):
                return f"{bracket}{items[0]}{close} * {len(items)}"
            # Multi-line format; items on separate lines.
            inner = ",\n".join(items)
            return bracket + "\n" + inner + ",\n" + close
    return _format_non_tensor_for_context(val)


def _get_param_names(func: Callable) -> list[str] | None:
    """Try to get positional parameter names from a function signature.

    Returns None if the signature cannot be inspected (e.g., C builtins).
    """
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]


def _classify_args(args: tuple, kwargs: dict) -> _ArgInfo:
    """Classify all tensor arguments in a single pass.

    Iterates every tensor in ``args`` and ``kwargs`` (skipping ``out=``),
    collecting ``get_local_type`` for each.  Untyped tensors contribute
    ``{}`` (unknown on all axes).

    Also records the original call arguments so error formatting can be
    deferred until a type error actually occurs.

    Args:
        args: Positional arguments (post-Scalar-unwrapping).
        kwargs: Keyword arguments (post-Scalar-unwrapping).
    """
    tensor_types: list[LocalSpmdType] = []
    raw_entries: list[_RawArgEntry] = []

    def _classify(t: torch.Tensor, location: str) -> None:
        lt = get_local_type(t)
        tensor_types.append(lt)

    for i, arg in enumerate(args):
        for t in _iter_tensors_in(arg):
            _classify(t, f"args[{i}]")
        raw_entries.append(_RawArgEntry(None, arg))
    for key, v in kwargs.items():
        if key == "out":
            continue
        for t in _iter_tensors_in(v):
            _classify(t, key)
        raw_entries.append(_RawArgEntry(key, v))

    return _ArgInfo(
        tensor_types,
        raw_entries,
    )


def _format_operator_context(
    func: Callable,
    raw_entries: list[_RawArgEntry],
) -> str:
    """Format an operator context string for error messages.

    Produces a multi-line block showing all arguments (tensor and
    non-tensor) like::

        In all_gather_into_tensor(
          output_tensor: f32[64] {},
          input_tensor: f32[32] {},
          group: DP,
        )

    Positional args use parameter names when available from the function
    signature, otherwise they fall back to ``args[i]``. Keyword args use
    their key name directly. ProcessGroup arguments are shown by their
    mesh axis name. Lists of tensors are shown with each element
    formatted inline.
    """
    #   4 spaces = arg indent (items in the function call)
    #   6 spaces = nested indent (items inside a list/tuple arg)
    _ARG_INDENT = "    "
    _NESTED_INDENT = "      "

    op_name = getattr(func, "__name__", repr(func))
    if not raw_entries:
        return f"  In {op_name}()"
    param_names = _get_param_names(func)
    parts = []
    positional_index = 0
    for entry in raw_entries:
        formatted = _format_arg_for_context(entry.value)
        if entry.location is not None:
            raw = f"{entry.location}: {formatted}"
        else:
            if param_names is not None and positional_index < len(param_names):
                label = param_names[positional_index]
            else:
                label = f"args[{positional_index}]"
            positional_index += 1
            raw = f"{label}: {formatted}"
        if "\n" not in raw:
            parts.append(f"{_ARG_INDENT}{raw}")
        else:
            # Multi-line value (e.g., list of tensors).  First line stays
            # at arg indent, middle lines at nested indent, last line
            # (closing bracket) back at arg indent.
            lines = raw.split("\n")
            result = f"{_ARG_INDENT}{lines[0]}"
            for line in lines[1:-1]:
                result += f"\n{_NESTED_INDENT}{line}"
            result += f"\n{_ARG_INDENT}{lines[-1]}"
            parts.append(result)
    joined = ",\n".join(parts)
    return f"  In {op_name}(\n{joined},\n  )"


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _collect_scalar_types(  # noqa: C901
    tensor_types: list[LocalSpmdType],
    original_args: tuple,
    original_kwargs: dict,
    spec: _OpSpec,
) -> list[LocalSpmdType]:
    """Append scalar types to tensor types based on op spec positions.

    Numeric scalars at declared tensor-input positions are included with
    ``_Scalar`` on all known mesh axes.  ``Scalar`` wrapper objects use their
    exact SPMD type.

    Additionally, ``Scalar`` wrapper objects at *any* argument position
    (not just tensor-input positions) participate in type inference.  This
    handles cases like ``narrow(tensor, dim, Scalar(V), Scalar(V))`` where
    the start/length are not tensor args but still carry rank-dependent
    values.  Plain numeric scalars at non-tensor-arg positions are ignored
    (they are structural parameters like ``dim``).

    Returns a new list with scalar types appended (or ``tensor_types``
    unchanged if no scalars are found).

    Args:
        tensor_types: Types already collected from typed tensors.
        original_args: Positional arguments (pre-Scalar-unwrapping, so
            ``Scalar`` objects are visible).
        original_kwargs: Keyword arguments (pre-Scalar-unwrapping).
        spec: Op specification declaring which positions are tensor inputs.
    """
    all_axes: set[DeviceMeshAxis] = set()
    for typ in tensor_types:
        all_axes.update(typ.keys())
    if not all_axes:
        return tensor_types

    scalar_type: LocalSpmdType = {axis: _Scalar for axis in all_axes}
    extra: list[LocalSpmdType] = []

    def _check(val: object) -> None:
        if isinstance(val, Scalar):
            extra.append(get_local_type(val))
        elif _is_numeric_scalar(val):
            extra.append(scalar_type)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Scalar):
                    extra.append(get_local_type(item))
                elif _is_numeric_scalar(item):
                    extra.append(scalar_type)

    def _check_scalar_only(val: object) -> None:
        """Check for Scalar wrappers only (ignore plain numeric scalars)."""
        if isinstance(val, Scalar):
            extra.append(get_local_type(val))
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Scalar):
                    extra.append(get_local_type(item))

    # Collect from declared tensor-input positions (Scalar + plain numeric).
    tensor_arg_positions: set[int] = set(spec.tensor_args)
    for i in spec.tensor_args:
        if i < len(original_args):
            _check(original_args[i])
    if spec.tensor_varargs_from is not None:
        for i in range(spec.tensor_varargs_from, len(original_args)):
            tensor_arg_positions.add(i)
            _check(original_args[i])
    for name in spec.tensor_kwargs:
        if name in original_kwargs:
            _check(original_kwargs[name])

    # Collect Scalar wrappers from non-tensor-arg positions.  Plain numeric
    # scalars here are structural (e.g. dim) and don't carry SPMD types.
    for i, arg in enumerate(original_args):
        if i not in tensor_arg_positions:
            _check_scalar_only(arg)
    for name, val in original_kwargs.items():
        if name not in spec.tensor_kwargs:
            _check_scalar_only(val)

    if extra:
        return list(tensor_types) + extra
    return tensor_types


def _get_mutated_tensors(  # noqa: C901
    func: Callable,
    args: tuple,
    kwargs: dict,
    result: object,
) -> list[torch.Tensor]:
    """Detect tensors being mutated/written-to by this operation.

    Uses three heuristics (any match is sufficient):
    1. An input tensor is identical to the result (result is args[i]).
    2. The function name has a trailing underscore (PyTorch in-place convention),
       excluding dunder methods but including __iadd__ etc.
    3. An out= keyword argument was provided.

    Returns:
        List of mutated tensors (deduplicated by identity).
    """
    mutated: list[torch.Tensor] = []
    seen_ids: set[int] = set()

    def _add(t: torch.Tensor) -> None:
        tid = id(t)
        if tid not in seen_ids:
            seen_ids.add(tid)
            mutated.append(t)

    # (3) out= kwarg
    out = kwargs.get("out")
    if out is not None:
        if isinstance(out, torch.Tensor):
            _add(out)
        elif isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    _add(item)

    # (1) Identity: result is one of the input tensors
    if isinstance(result, torch.Tensor):
        for arg in args:
            if isinstance(arg, torch.Tensor) and result is arg:
                _add(result)
                break

    # (2) Trailing underscore (fallback when identity/out didn't trigger).
    # Catches method-style in-place ops (add_, zero_) and dunder in-place
    # ops (__iadd__, __imul__). PASSTHROUGH functions (requires_grad_, etc.)
    # never reach this code path.
    if not mutated:
        func_name = getattr(func, "__name__", "")
        is_inplace_name = (
            func_name.endswith("_") and not func_name.startswith("__")
        ) or (
            func_name.startswith("__i")
            and func_name.endswith("__")
            and len(func_name) > 5
        )
        if is_inplace_name and args and isinstance(args[0], torch.Tensor):
            _add(args[0])

    return mutated


def _validate_mutation_types(
    func: Callable,
    mutated_tensors: list[torch.Tensor],
    output_type: LocalSpmdType,
) -> None:
    """Validate that mutated tensors' existing SPMD types match the output type.

    For mutating/out operations, the operation writes into an existing tensor.
    The output SPMD type must match the mutated tensor's existing type on every
    axis; otherwise the mutation would silently change the tensor's SPMD type,
    which is unsound (other references to the tensor still expect the old type).

    Args:
        func: The torch function (for error messages).
        mutated_tensors: Tensors being mutated by this operation.
        output_type: The inferred output SPMD type.

    Raises:
        SpmdTypeError: If any mutated tensor's type conflicts with output_type.
    """
    func_name = getattr(func, "__name__", repr(func))
    for t in mutated_tensors:
        existing = get_local_type(t)
        for axis, new_typ in output_type.items():
            old_typ = existing.get(axis)
            if old_typ is not None and old_typ is not new_typ:
                raise SpmdTypeError(
                    f"{func_name}: in-place/out operation would change "
                    f"SPMD type on axis {format_axis(axis)} from {old_typ} "
                    f"to {new_typ}. In-place and out= operations cannot "
                    f"change a tensor's SPMD type."
                    # TODO: For in-place ops that support autograd, we may
                    # want to allow the type to change (mutating the SPMD
                    # type of the out argument).
                )


def _set_result_type(result: object, output_type: LocalSpmdType) -> None:
    """Set SPMD types on the result tensor(s).

    Handles single tensors, flat list/tuple of tensors, and arbitrary nested
    structures (e.g., NamedTuples from ops like torch.linalg.lu_factor).
    The common cases (single tensor, flat sequence) are checked first to
    avoid the overhead of pytree flattening.
    """

    def _apply(t: torch.Tensor) -> None:
        _set_local_type(t, output_type)

    if isinstance(result, torch.Tensor):
        _apply(result)
        return
    if isinstance(result, (list, tuple)):
        all_flat = True
        for item in result:
            if isinstance(item, torch.Tensor):
                _apply(item)
            elif isinstance(item, (list, tuple, dict)):
                all_flat = False
        if all_flat:
            return
    # Fall back to pytree for nested structures.
    flat, _ = torch.utils._pytree.tree_flatten(result)
    for item in flat:
        if isinstance(item, torch.Tensor):
            _apply(item)


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
            if isinstance(val, torch.Tensor):
                if i in fixed_positions:
                    fixed_types.append(get_local_type(val))
                else:
                    free_types.append(get_local_type(val))
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, torch.Tensor):
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
    # plus any scalar types that were appended by _collect_scalar_types.
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
    invariant_to_replicate: {"src": I, "dst": R},
}


# =============================================================================
# TorchFunctionMode for SPMD Type Tracking
# =============================================================================

# Functions that are not tensor math -- autograd bookkeeping, metadata queries,
# etc.  These go through __torch_function__ but should not trigger type
# inference or strict-mode annotation checks.
#
# This set is intentionally exhaustive: any function that reaches
# __torch_function__ and is NOT in this set will be type-checked.  This
# ensures that unrecognized operations (e.g. raw torch.distributed
# collectives) are caught rather than silently bypassing the checker.
#
# Note: property accesses like .shape/.ndim are handled by the __get__
# check in __torch_function__, and .stride() is in PyTorch's
# get_ignored_functions() so it never reaches __torch_function__ at all.
_PASSTHROUGH = {
    # Autograd bookkeeping
    # TODO: Actually, backward probably can support type checking; in
    # particular we could put annotations on all of the resulting grad
    # tensors based on the typing of the leaf tensor.
    torch.Tensor.backward,
    torch.Tensor.requires_grad_,
    torch.Tensor.retain_grad,
    # Metadata queries (return non-tensor values)
    torch.Tensor.dim,
    torch.Tensor.element_size,
    torch.Tensor.get_device,
    torch.Tensor.is_complex,
    torch.Tensor.is_contiguous,
    torch.Tensor.is_floating_point,
    torch.Tensor.nelement,
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.untyped_storage,
    torch.numel,
}


# Import shared state from _state module to avoid circular dependencies
import sixlib.spmd_types._state as _state  # noqa: E402
from sixlib.spmd_types._raw_dist import RAW_DIST_RULES  # noqa: E402
from sixlib.spmd_types._state import is_type_checking  # noqa: E402, F401

# =============================================================================
# Autograd Function.apply monkeypatch
# =============================================================================

# Set of autograd.Function subclasses whose apply is known to be local-only
# (i.e., each output element depends only on the corresponding input elements
# so the standard element-wise type propagation rule is safe).
_LOCAL_AUTOGRAD_FUNCTIONS: set[type] = set()

# Set of autograd.Function subclasses registered with a typecheck_forward
# staticmethod.  When .apply() is intercepted by __torch_function__,
# typecheck_forward is called INSTEAD of .apply().  It should assert_type()
# on inputs, call .apply() to execute, and assert_type() on the output.
_TYPECHECK_AUTOGRAD_FUNCTIONS: set[type] = set()


def register_local_autograd_function(cls: type) -> type:
    """Register an autograd.Function subclass as local-only for SPMD type checking.

    Local-only means the function's forward operates element-wise (or more
    generally, does not rearrange data across the tensor in a way that would
    change its sharding type).  It must NOT perform any collectives or
    cross-device communication.  For functions that do, use
    :func:`register_autograd_function` with a ``typecheck_forward`` method instead.

    Registered functions get the standard local type propagation rule when
    type checking is active:

    - Inputs may freely mix R and V types; the output is R unless any input
      is V, in which case it is V.
    - All-I inputs produce I outputs.
    - R/V and I cannot be mixed.
    - P is forbidden.

    Unregistered autograd functions that reach the type checker will leave
    their outputs untyped (or raise in strict mode), since the checker
    cannot know whether the function is safe for automatic type propagation.

    Can be used as a decorator::

        @register_local_autograd_function
        class MyOp(torch.autograd.Function):
            ...
    """
    _LOCAL_AUTOGRAD_FUNCTIONS.add(cls)
    return cls


def register_autograd_function(cls: type) -> type:
    """Register an autograd.Function subclass with a custom typecheck method.

    Use this for autograd functions that perform collectives or have
    non-trivial type transformations where the default local-only rule
    would produce incorrect output types.

    The class must define a ``typecheck_forward`` staticmethod that receives
    the same positional/keyword arguments as ``.apply()``.  Inside, it should
    call ``assert_type`` on inputs, call ``.apply()`` to run the function,
    then call ``assert_type`` on the output.  This is symmetric with
    ``typecheck_forward`` on ``nn.Module`` subclasses in llama4x::

        @register_autograd_function
        class MyCollectiveOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def typecheck_forward(x, y):
                assert_type(x, {pg: S(-1)})
                out = MyCollectiveOp.apply(x, y)
                assert_type(out, {pg: R})
                return out

            @staticmethod
            def backward(ctx, g):
                return g, g
    """
    if not callable(getattr(cls, "typecheck_forward", None)):
        raise TypeError(
            f"{cls.__name__} must define a typecheck_forward staticmethod "
            f"when using @register_autograd_function"
        )
    _TYPECHECK_AUTOGRAD_FUNCTIONS.add(cls)
    return cls


# The original C++ descriptor for autograd.Function.apply, saved when the
# patch is installed and restored when the patch is removed.
_orig_autograd_apply = None


class _AutogradApplyDescriptor:
    """Wraps autograd.Function.apply to dispatch through __torch_function__.

    We wrap Function.apply (the Python classmethod that handles
    vmap/functorch dispatch), NOT _FunctionBase.apply (the raw C++
    apply).  This preserves the vmap-aware code path so that autograd
    functions with ``generate_vmap_rule=True`` continue to work under
    vmap while SpmdTypeMode is active.
    """

    def __init__(self, orig):
        self._orig = orig  # Original Function.apply classmethod

    def __get__(self, obj, cls=None):
        if cls is None and obj is not None:
            cls = type(obj)
        # Bind original apply to this specific Function subclass
        orig_method = self._orig.__get__(obj, cls)

        if not _state.is_type_checking():
            return orig_method

        def wrapper(*args, **kwargs):
            tensors = tuple(
                a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)
            )
            if tensors and has_torch_function(tensors):
                return handle_torch_function(orig_method, tensors, *args, **kwargs)
            return orig_method(*args, **kwargs)

        return wrapper


def _install_autograd_apply_patch():
    """Replace Function.apply with a descriptor that dispatches through __torch_function__."""
    global _orig_autograd_apply
    # Save the original classmethod from Function (NOT _FunctionBase).
    # Function.apply is a Python classmethod that handles vmap/functorch
    # dispatch; _FunctionBase.apply is the raw C++ apply that does not.
    _orig_autograd_apply = torch.autograd.Function.__dict__["apply"]
    torch.autograd.Function.apply = _AutogradApplyDescriptor(_orig_autograd_apply)


def _remove_autograd_apply_patch():
    """Restore the original Function.apply classmethod."""
    global _orig_autograd_apply
    if _orig_autograd_apply is not None:
        torch.autograd.Function.apply = _orig_autograd_apply
    _orig_autograd_apply = None


def _get_autograd_function_class(func) -> type | None:
    """Return the autograd.Function subclass if func is a bound apply method, else None."""
    if getattr(func, "__name__", None) != "apply":
        return None
    cls = getattr(func, "__self__", None)
    if cls is None or not isinstance(cls, type):
        return None
    if issubclass(cls, torch.autograd.Function):
        return cls
    return None


class _SpmdTypeMode(torch.overrides.TorchFunctionMode):
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
        strict_mode: Controls the strictness of type checking.

            - ``"permissive"``: allows mixing annotated and unannotated
              tensor operands without error.  Useful during incremental
              annotation of an existing codebase.

            - ``"strict"`` (default): raises ``SpmdTypeError`` when a
              regular torch op mixes typed and untyped tensor operands.
              Ops with no typed tensor inputs (like ``torch.zeros``)
              produce tensors with ``{}`` type (typed but unknown on
              all axes).  Combining a ``{}`` tensor with a typed tensor
              that has axes raises ``SpmdTypeError`` because the ``{}``
              tensor is missing those axes.  Call ``assert_type()`` to
              assign axes before combining.
    """

    def __init__(
        self,
        *,
        strict_mode: Literal["permissive", "strict"] = "strict",
    ):
        super().__init__()
        self._strict = strict_mode == "strict"
        self._disabled = False

    def __enter__(self):
        assert getattr(_state._tls, "active_mode", None) is None, (
            "_SpmdTypeMode must only be entered via typecheck()"
        )
        # FIXME: the autograd patch mutates a global class attribute
        # (torch.autograd.Function.apply).  If multiple threads enter/exit
        # concurrently, one thread may remove the patch while another is
        # still active.  This will need a lock + refcount when we support
        # multi-threaded type checking.
        _install_autograd_apply_patch()

        # Patch vmap so BatchedTensors carry SPMD type annotations.
        import sixlib.spmd_types._vmap as _vmap

        _vmap.install()

        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = super().__exit__(exc_type, exc_val, exc_tb)
        _remove_autograd_apply_patch()

        import sixlib.spmd_types._vmap as _vmap

        _vmap.uninstall()

        return result

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}
        # Paused via no_typecheck(): run without type checking.
        if self._disabled:
            return func(*args, **kwargs)

        # Property access (e.g. .grad, .data, .shape) -- not tensor math.
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # Autograd bookkeeping, metadata queries -- not tensor math.
        if func in _PASSTHROUGH:
            return func(*args, **kwargs)

        # Outer try/except: apply traceback filtering to SpmdTypeError.
        # This wraps everything after the early returns so that context
        # enrichment (inner except) runs first, then filtering strips
        # internal frames from the traceback the user sees.
        try:
            return self._typecheck_core(func, types, args, kwargs)
        except SpmdTypeError as e:
            _filter_and_reraise(e)
            raise

    def _typecheck_core(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}
        # Unwrap Scalar objects to raw values for the actual function call,
        # but keep the original args for type collection.
        original_args, original_kwargs = args, kwargs
        args, kwargs = _unwrap_args(args, kwargs)

        # Typecheck-registered autograd function: dispatch to typecheck_forward
        # INSTEAD of executing func.  The mode is popped off the
        # TorchFunctionMode stack during __torch_function__ dispatch (via
        # _pop_mode_temporarily in torch/overrides.py), so .apply() called
        # inside typecheck_forward executes normally without re-entering
        # this method.
        autograd_cls = _get_autograd_function_class(func)
        if autograd_cls is not None and autograd_cls in _TYPECHECK_AUTOGRAD_FUNCTIONS:
            return autograd_cls.typecheck_forward(*args, **kwargs)

        # DTensor tracks its own placement metadata; the SPMD type checker
        # should not interfere with DTensor operations.  Registered autograd
        # functions (e.g. _ToTorchTensor) are already handled above.
        if any(issubclass(t, DTensor) for t in types):
            return func(*args, **kwargs)

        # Run the function, then type-check the result below.
        # Raw torch.distributed collectives are already type-checked above;
        # SPMD collectives and regular ops are checked after execution.
        result = func(*args, **kwargs)

        # =============================================================
        # Classification phase: single pass over all tensor args.
        # =============================================================
        info = _classify_args(args, kwargs)

        # =============================================================
        # Type checking phase.
        #
        # Every tensor's type is collected via get_local_type (untyped
        # tensors contribute {} -- unknown on all axes).  The per-axis
        # inference rules in infer_output_type will raise SpmdTypeError
        # when incompatible types are combined.
        # =============================================================
        try:
            if func in _SPMD_FUNCTION_DEFAULTS:
                x = args[0]
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = normalize_axis(args[1] if len(args) > 1 else kwargs["axis"])
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                # Decay Shard to Varying for local SPMD type checking.
                # S(i) is a global SPMD refinement; locally it behaves as V.
                if isinstance(src, Shard):
                    src = V
                if isinstance(dst, Shard):
                    dst = V

                # Validate input type on the axis matches src.
                # Special case: reductions (all_reduce, reduce_scatter) accept
                # V when src=P, implicitly reinterpreting V as P.  The runtime
                # in _collectives.py already handles this conversion.
                local_type = get_local_type(x)
                if axis in local_type:
                    input_type = local_type[axis]
                    if src is not None and input_type != src:
                        # Allow V -> P implicit cast for reductions
                        if not (src is P and input_type is V):
                            raise SpmdTypeError(
                                f"{func.__name__}: expected input type {src} on axis "
                                f"{format_axis(axis)}, got {input_type}"
                            )
                elif axis.size() == 1:
                    pass  # singleton -- skip type validation
                elif _state.is_strict():
                    raise SpmdTypeError(
                        f"{func.__name__}: tensor has no type for axis "
                        f"{format_axis(axis)}. Use assert_type() to annotate "
                        f"the tensor on this axis before calling collectives."
                    )
                # else: permissive mode -- skip validation for this axis,
                # but still propagate types on other axes below.

                # Build output types: copy all axes from input, override this axis.
                # Always set dst even if axis was previously missing -- we know
                # the post-collective type.
                output_type = local_type.copy()
                if dst is not None:
                    output_type[axis] = dst
                _set_local_type(result, output_type)

                if _TRACE:
                    _trace_op(func, info.tensor_types, output_type)

            elif func in RAW_DIST_RULES:
                # Raw torch.distributed collective (e.g. all_gather_into_tensor).
                # The rule functions validate types internally.
                RAW_DIST_RULES[func](func, args, kwargs)

                if _TRACE:
                    _trace_op(func, info.tensor_types, None)

            elif (
                autograd_cls is not None
                and autograd_cls not in _LOCAL_AUTOGRAD_FUNCTIONS
            ):
                # Unregistered autograd Function: unknown semantics, cannot
                # safely assume element-wise behaviour.
                if _state.is_strict():
                    raise SpmdTypeError(
                        f"{autograd_cls.__name__}.apply: autograd Function "
                        f"is not registered for SPMD type "
                        f"checking. Use "
                        f"{register_local_autograd_function.__name__}("
                        f"{autograd_cls.__name__}) to mark it as safe "
                        f"for type propagation, and provide "
                        f"register_autograd_function("
                        f"{autograd_cls.__name__}) for a custom "
                        f"typecheck_forward."
                    )
                return result

            else:
                # Local op (regular op or registered local autograd Function):
                # infer output type from tensor types + scalars.
                if not info.tensor_types:
                    # No typed tensor inputs (no tensors at all).
                    # Set {} (typed but unknown on all axes).
                    _set_result_type(result, {})
                    return result

                spec = _OP_REGISTRY.get(func)
                input_types_list = list(info.tensor_types)
                if spec is not None:
                    input_types_list = _collect_scalar_types(
                        input_types_list, original_args, original_kwargs, spec
                    )

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

                # Validate mutation safety for in-place/out operations.
                mutated = _get_mutated_tensors(func, args, kwargs, result)
                if mutated:
                    _validate_mutation_types(func, mutated, output_type)

                _set_result_type(result, output_type)
                out = original_kwargs.get("out")
                if out is not None:
                    _set_result_type(out, output_type)

                if _TRACE:
                    _trace_op(func, info.tensor_types, output_type)

        except SpmdTypeError as e:
            if e.context is None:
                e.context = _format_operator_context(func, info.raw_entries)
            raise

        return result


@contextmanager
def typecheck(
    strict_mode: Optional[Literal["permissive", "strict"]] = None,
):
    """Context manager that activates SPMD type checking.

    If type checking is already active, temporarily overrides ``strict_mode`` on
    the existing mode (reentrant no-op for the mode itself).  If not active,
    creates and enters a new ``_SpmdTypeMode``.

    Args:
        strict_mode: Controls the strictness of type checking.  See
            ``_SpmdTypeMode`` for details on each mode.  When ``None``
            (the default), inherits the current strict mode if reentrant,
            or uses ``"strict"`` when creating a fresh mode.
    """
    existing = getattr(_state._tls, "active_mode", None)
    if existing is not None:
        old_strict = existing._strict
        old_disabled = existing._disabled
        if strict_mode is not None:
            existing._strict = strict_mode == "strict"
        existing._disabled = False
        try:
            yield
        finally:
            existing._strict = old_strict
            existing._disabled = old_disabled
    else:
        if strict_mode is None:
            strict_mode = "strict"
        mode = _SpmdTypeMode(strict_mode=strict_mode)
        with mode:
            _state._set_active_mode(mode)
            try:
                yield
            finally:
                _state._clear_active_mode()


@contextmanager
def no_typecheck():
    """Temporarily disable type checking on this thread (like ``no_grad``)."""
    mode = getattr(_state._tls, "active_mode", None)
    if mode is not None:
        old_disabled = mode._disabled
        mode._disabled = True
        try:
            yield
        finally:
            mode._disabled = old_disabled
    else:
        yield


class _SpmdTypeBackwardCompatibleMode:
    """Backwards-compatible wrapper around ``typecheck()``.

    Exported as ``SpmdTypeMode`` for existing callers that do::

        with SpmdTypeMode():
            ...

    New code should use ``typecheck()`` instead.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        self._cm = typecheck(**self._kwargs)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._cm.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# DTensor <-> local tensor boundary typecheck_forward implementations
# =============================================================================


def _placements_to_local_type(mesh, placements, grad_placements):
    """Convert DTensor placements to an SPMD local type dict."""
    from sixlib.spmd_types._dtensor import dtensor_placement_to_spmd_type

    local_type: LocalSpmdType = {}
    for i, placement in enumerate(placements):
        pg = mesh.get_group(i)
        axis = normalize_axis(pg)
        grad_pl = grad_placements[i] if i < len(grad_placements) else None
        spmd_type = dtensor_placement_to_spmd_type(placement, grad_pl)
        # S(i) decays to V for local type storage.
        if isinstance(spmd_type, Shard):
            spmd_type = V
        local_type[axis] = spmd_type
    return local_type


def _typecheck_forward_to_torch_tensor(*args, **kwargs):
    """typecheck_forward for _ToTorchTensor: run apply, then assert_type."""
    # _ToTorchTensor.apply(input_dtensor, grad_placements)
    input_dtensor = args[0]
    assert isinstance(input_dtensor, DTensor), (
        f"_ToTorchTensor input must be a DTensor, got {type(input_dtensor)}"
    )
    grad_placements = args[1] if len(args) > 1 else None
    if grad_placements is None:
        grad_placements = _normalize_placements_for_grad(input_dtensor.placements)
    result = _ToTorchTensor.apply(*args, **kwargs)
    local_type = _placements_to_local_type(
        input_dtensor.device_mesh, input_dtensor.placements, grad_placements
    )
    assert_type(result, local_type)
    return result


def _typecheck_forward_from_torch_tensor(*args, **kwargs):
    """typecheck_forward for _FromTorchTensor: validate types, then run apply."""
    # _FromTorchTensor.apply(input, device_mesh, placements, run_check,
    #                        shape, stride, grad_placements)
    input_tensor = args[0]
    device_mesh = args[1]
    placements = args[2]
    grad_placements = args[6] if len(args) > 6 else None
    if grad_placements is None:
        grad_placements = _normalize_placements_for_grad(placements)
    expected_type = _placements_to_local_type(device_mesh, placements, grad_placements)
    assert_type(input_tensor, expected_type)
    return _FromTorchTensor.apply(*args, **kwargs)


_ToTorchTensor.typecheck_forward = staticmethod(_typecheck_forward_to_torch_tensor)
_FromTorchTensor.typecheck_forward = staticmethod(_typecheck_forward_from_torch_tensor)
_TYPECHECK_AUTOGRAD_FUNCTIONS.add(_ToTorchTensor)
_TYPECHECK_AUTOGRAD_FUNCTIONS.add(_FromTorchTensor)
