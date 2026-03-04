"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, overload

import torch
import torch.overrides
from sixlib.spmd_types._collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
)
from sixlib.spmd_types._local import convert, invariant_to_replicate, reinterpret
from sixlib.spmd_types._scalar import _unwrap_args, Scalar
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    I,
    LocalSpmdType,
    P,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    R,
    Shard,
    SpmdTypeError,
    V,
)

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

# Attribute name for storing SPMD types on tensors
_LOCAL_TYPE_ATTR = "_local_type"


def _validate(type: LocalSpmdType) -> LocalSpmdType:
    """Validate a LocalSpmdType.

    Validates that all values are valid local SPMD types (R, I, V, or P).
    Shard types are not valid local SPMD types -- they are used as arguments to
    collective operations but must not be stored on tensors.

    Args:
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P).
    """
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
            raise TypeError(
                f"Expected PerMeshAxisLocalSpmdType (R, I, V, or P) on axis "
                f"{format_axis(axis)}, got {typ!r}"
            )
    return dict(type)


def has_local_type(value: torch.Tensor | Scalar) -> bool:
    """Return True if the tensor or Scalar has SPMD type annotations.

    Args:
        value: The tensor or Scalar to check for SPMD type annotations.
    """
    return hasattr(value, _LOCAL_TYPE_ATTR)


def get_local_type(value: torch.Tensor | Scalar) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor or Scalar.

    Args:
        value: The tensor or Scalar to retrieve SPMD type annotations from.

    Raises:
        AttributeError: If the object has no SPMD type annotations.
            Use ``has_local_type`` to check first.
    """
    result = getattr(value, _LOCAL_TYPE_ATTR, None)
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
    setattr(tensor, _LOCAL_TYPE_ATTR, _validate(type))
    return tensor


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


def assert_type(  # noqa: C901
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes,
    partition_spec: PartitionSpec | None = None,
) -> torch.Tensor:
    """
    Assert or set the SPMD type on a tensor.

    If the tensor has no SPMD type, sets it.
    If the tensor already has an SPMD type, checks it equals the provided type.

    If a mesh axis is omitted from type and not mentioned in partition_spec,
    it is assumed that the tensor varies over that mesh axis without a global
    SPMD type.

    Mesh axes that are R, I, or P must be specified in ``type``, even if they
    could be inferred from ``partition_spec``, because it would be ambiguous
    whether they are Replicate or Invariant.

    As syntax sugar, ``S(i)`` can be used in ``type`` to indicate that a mesh
    axis shards tensor dimension ``i``.  This is equivalent to omitting the axis
    from ``type`` and including it in ``partition_spec``.  However, ``S(i)``
    entries cannot be mixed with an explicit ``partition_spec``, and we reject
    if two axes shard the same tensor dimension via ``S(i)`` since the order
    is ambiguous.

    Returns the tensor for chaining.

    Args:
        tensor: The tensor to assert or set SPMD type on.
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.
            Accepts R, I, V, P, or S(i) (as syntax sugar for partition_spec).
        partition_spec: Optional PartitionSpec describing how tensor dimensions
            map to mesh axes for Varying dimensions.

    Raises:
        TypeError: If type contains invalid type objects or if S(i) is mixed
            with an explicit partition_spec
        SpmdTypeError: If partition_spec length doesn't match tensor ndim, or
            if a partition_spec axis conflicts with a non-Varying type in type
        AssertionError: If existing type doesn't match provided type
    """
    # Separate S(i) entries from R/I/V/P entries
    local_type: LocalSpmdType = {}
    shard_entries: dict[DeviceMeshAxis, Shard] = {}
    for axis, typ in type.items():
        if isinstance(typ, Shard):
            shard_entries[axis] = typ
        else:
            local_type[axis] = typ

    # Handle S(i) sugar
    if shard_entries:
        if partition_spec is not None:
            raise TypeError(
                "Cannot use S(i) in type and an explicit partition_spec at the "
                "same time. Use one or the other."
            )
        # Check for out-of-bounds and duplicate tensor dims
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
        # S(i) axes are implicitly V -- store V explicitly
        for axis in shard_entries:
            local_type[axis] = V

    # Validate partition_spec
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
                # partition_spec axes are implicitly V -- store V explicitly
                local_type.setdefault(axis, V)

    canonical = _validate(local_type)
    if not has_local_type(tensor):
        return _set_local_type(tensor, canonical)

    # existing is already validated (_set_local_type validates on store)
    existing = get_local_type(tensor)
    if existing != canonical:
        raise AssertionError(
            f"SPMD type mismatch: tensor has {existing}, expected {canonical}"
        )
    return tensor


def assert_local_type(tensor: torch.Tensor, type: PerMeshAxisSpmdTypes) -> torch.Tensor:
    """Deprecated: use ``assert_type`` instead."""
    return assert_type(tensor, type)


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
            if axis not in typ:
                raise SpmdTypeError(
                    f"Operand missing axis {format_axis(axis)}. "
                    f"Operand has axes {set(typ.keys())}, "
                    f"but the union of all operand axes is {all_axes}. "
                    f"All operands must be annotated on the same set of mesh axes."
                )
            axis_types.append(typ[axis])

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


def _iter_tensor_args(args: tuple, kwargs: dict):
    """Yield all tensor arguments from args and kwargs.

    Handles direct tensors and flat list/tuple of tensors (common cases) with
    a fast path, and falls back to pytree flattening for arbitrary nesting.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    for arg in args:
        yield from _iter_tensors_in(arg)
    for key, v in kwargs.items():
        if key == "out":
            continue
        yield from _iter_tensors_in(v)


def _check_all_typed(func, args: tuple, kwargs: dict) -> None:
    """Raise ``SpmdTypeError`` if any operand is unannotated.

    Called once at the top of the regular-op path in strict mode so that
    ``_collect_tensor_types`` itself stays simple and unconditional.

    Uses ``has_local_type`` rather than truthiness of the types dict because
    a tensor's type dict is an implementation detail -- ``has_local_type`` is
    the canonical way to check whether a tensor has been annotated.

    Args:
        func: The torch function being called.
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
    """
    untyped: list[str] = []
    for i, arg in enumerate(args):
        for t in _iter_tensors_in(arg):
            if not has_local_type(t):
                untyped.append(f"args[{i}] (shape={tuple(t.shape)}, dtype={t.dtype})")
    for key, v in kwargs.items():
        if key == "out":
            continue
        for t in _iter_tensors_in(v):
            if not has_local_type(t):
                untyped.append(
                    f"kwargs[{key!r}] (shape={tuple(t.shape)}, dtype={t.dtype})"
                )
    if untyped:
        func_name = getattr(func, "__name__", repr(func))
        listing = "\n  ".join(untyped)
        raise SpmdTypeError(
            f"Strict mode: {len(untyped)} unannotated tensor(s) "
            f"in operation {func_name}:\n  {listing}\n"
            f"All tensor operands must be annotated. "
            f"Use SpmdTypeMode(strict=False) if you want partial type checking."
        )


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _collect_input_types(  # noqa: C901
    args: tuple, kwargs: dict, spec: _OpSpec | None = None
) -> list[LocalSpmdType]:
    """Collect SPMD types from tensor args and scalar args.

    When a spec is available, numeric scalars at declared tensor-input
    positions are included with _Scalar on all known mesh axes.  Without a
    spec we cannot distinguish scalar operands from non-tensor parameters
    (like ``dim``), so only tensor types are collected.

    We assume a Python scalar is the same value on all ranks and carries no
    gradient.  Rather than eagerly picking R or I, we assign _Scalar -- a typeless
    sentinel analogous to "weak" scalars in dtype promotion.  The inference
    engine filters _Scalar when determining the output type, so scalars
    never force a choice between R and I.

    Args:
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
        spec: Optional op specification for detecting scalar tensor positions.
    """
    result: list[LocalSpmdType] = []
    for t in _iter_tensor_args(args, kwargs):
        if has_local_type(t):
            result.append(get_local_type(t))

    # _Scalar is typeless (doesn't force R vs I), so it is safe to collect
    # for all linearities -- the old LINEAR-only gate was only needed when
    # scalars were eagerly assigned R.
    if result and spec is not None:
        all_axes: set[DeviceMeshAxis] = set()
        for typ in result:
            all_axes.update(typ.keys())
        if all_axes:
            scalar_type: LocalSpmdType = {axis: _Scalar for axis in all_axes}

            def _check(val: object) -> None:
                if isinstance(val, Scalar):
                    # Scalar wrapper: use its exact SPMD type.
                    result.append(get_local_type(val))
                elif _is_numeric_scalar(val):
                    result.append(scalar_type)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, Scalar):
                            result.append(get_local_type(item))
                        elif _is_numeric_scalar(item):
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
    """Set SPMD types on the result tensor(s).

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
from sixlib.spmd_types._state import is_type_checking  # noqa: E402, F401


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

    def __init__(self, strict: bool = True):
        super().__init__()
        self.strict = strict
        self._disabled = False

    @contextmanager
    def disable(self):
        """Temporarily disable type checking.

        Use this to run operations that should not be type-checked,
        such as per-rank callbacks in rank_map or value assertions
        on internal tensor data.
        """
        old = self._disabled
        self._disabled = True
        if not old:
            _state._spmd_type_mode_active -= 1
        try:
            yield
        finally:
            self._disabled = old
            if not old:
                _state._spmd_type_mode_active += 1

    def __enter__(self):
        _state._spmd_type_mode_active += 1
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _state._spmd_type_mode_active -= 1
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}

        # Paused: run without type checking.
        if self._disabled:
            return func(*args, **kwargs)

        # Property access (e.g. .grad, .data, .shape) -- not tensor math.
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # Autograd bookkeeping, metadata queries -- not tensor math.
        if func in _PASSTHROUGH:
            return func(*args, **kwargs)

        # Unwrap Scalar objects to raw values for the actual function call,
        # but keep the original args for type collection.
        original_args, original_kwargs = args, kwargs
        args, kwargs = _unwrap_args(args, kwargs)

        # Run the function first (catches runtime errors before type errors).
        # This means collectives execute before type checking, but that's fine:
        # type checking typically runs with a fake PG where comms are free.
        result = func(*args, **kwargs)

        # Strict mode: all tensor operands must be annotated (applies to both
        # SPMD collectives and regular torch ops).
        if self.strict:
            _check_all_typed(func, args, kwargs)

        if func in _SPMD_FUNCTION_DEFAULTS:
            # Special spmd collective/reinterpret/collect
            x = args[0]
            if has_local_type(x):
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = args[1] if len(args) > 1 else kwargs["axis"]
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                # Decay Shard to Varying for local SPMD type checking.
                # S(i) is a global SPMD refinement; locally it behaves as V.
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
            # Use original_args/original_kwargs so Scalar types are collected.
            spec = _OP_REGISTRY.get(func)
            input_types_list = _collect_input_types(
                original_args, original_kwargs, spec
            )
            if input_types_list:
                # Check for type-level decomposition first.
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
                out = original_kwargs.get("out")
                if out is not None:
                    _set_result_type(out, output_type)

        return result
