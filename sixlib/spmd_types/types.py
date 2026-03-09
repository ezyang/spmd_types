"""
SPMD type definitions for distributed tensor expressions.

This module provides:
- Per-mesh-axis local SPMD types (R, I, V, P, S)
- PartitionSpec for global SPMD
- TensorSharding for module boundary contracts
- Type aliases
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# =============================================================================
# Per-Mesh-Axis Local SPMD Type Enum
# =============================================================================


class PerMeshAxisLocalSpmdType(Enum):
    """
    Per-mesh-axis local SPMD type as an enum.

    Describes how a tensor is distributed across ranks on one axis of the
    device mesh, as well as how the gradients are distributed. The four
    values are: R (Replicate), I (Invariant), V (Varying), P (Partial).
    """

    R = "R"
    I = "I"  # noqa: E741
    V = "V"
    P = "P"

    def __repr__(self):
        return self.value

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        """Return the type that gradients have in the backward pass."""
        return _BACKWARD_TYPE[self]


_BACKWARD_TYPE = {
    PerMeshAxisLocalSpmdType.R: PerMeshAxisLocalSpmdType.P,
    PerMeshAxisLocalSpmdType.I: PerMeshAxisLocalSpmdType.I,
    PerMeshAxisLocalSpmdType.V: PerMeshAxisLocalSpmdType.V,
    PerMeshAxisLocalSpmdType.P: PerMeshAxisLocalSpmdType.R,
}


@dataclass(frozen=True)
class Shard:
    """
    A refinement of Varying that specifies the tensor is sharded on a particular
    dimension.

    While Varying (V) only says "each rank has different data" without
    specifying how ranks relate to a global tensor, Shard(dim) additionally
    says "the global tensor can be reconstructed by concatenating the local
    tensors along dimension ``dim``."  This is analogous to DTensor's
    ``Shard(dim)`` placement.

    This is not a true type (notice it doesn't inherit from PerMeshAxisLocalSpmdType)
    but it is accepted at any src/dst argument and, from a typing perspective,
    is equivalent to Varying.  However, it does change the semantics of collectives
    by switching from stack/unbind semantics (V) to concat/split semantics (S):

    - ``all_gather(src=V)``: stacks per-rank tensors on a new dim 0
      (output gains a dimension).
    - ``all_gather(src=S(i))``: concatenates per-rank tensors along dim ``i``
      (output has the same number of dimensions as each input).
    - ``reduce_scatter(dst=V)``: unbinds along dim 0 after reducing
      (output loses a dimension).
    - ``reduce_scatter(dst=S(i))``: splits along dim ``i`` after reducing
      (output has the same number of dimensions, but dim ``i`` shrinks).

    In practice, most users want concat/split (S) rather than stack/unbind (V),
    because tensors are typically sharded along an existing dimension.

    In global SPMD types, a per-mesh-axis Shard can also be used to manipulate
    the PartitionSpec in a mesh-oriented way, although the PartitionSpec is still
    the canonical way of representing this typing information.
    """

    dim: int

    def __repr__(self):
        return f"S({self.dim})"

    def backward_type(self) -> Shard:
        return self


# Single character aliases for ease of pattern matching
R = PerMeshAxisLocalSpmdType.R
I = PerMeshAxisLocalSpmdType.I  # noqa: E741
V = PerMeshAxisLocalSpmdType.V
P = PerMeshAxisLocalSpmdType.P
S = Shard  # S(i) creates a Shard with dim=i

# Backward compatibility aliases
Replicate = R
Invariant = I
Varying = V
Partial = P

# Type aliases
PerMeshAxisSpmdType = PerMeshAxisLocalSpmdType | Shard

# Axis identifier: either a mesh axis name (string) or a ProcessGroup directly
DeviceMeshAxis: TypeAlias = "str | ProcessGroup"

# PerMeshAxisSpmdTypes is the permissive input variant that also accepts S(i).
# Used in user-facing APIs like assert_type, where S(i) is syntax sugar
# for setting the partition_spec.
PerMeshAxisSpmdTypes: TypeAlias = "dict[DeviceMeshAxis, PerMeshAxisSpmdType]"

# LocalSpmdType maps axis identifiers to per-axis SPMD types (R, I, V, P only).
# This is the type stored on tensors; Shard is never stored.
LocalSpmdType: TypeAlias = "dict[DeviceMeshAxis, PerMeshAxisLocalSpmdType]"

# =============================================================================
# PartitionSpec for Global SPMD
# =============================================================================


class PartitionSpec(tuple):
    """
    A partition spec describes how tensor dimensions map to mesh axes.

    Each element corresponds to a tensor dimension and specifies zero, one, or
    multiple mesh axis names that shard that dimension. For example:
        - PartitionSpec('tp', None) means dim 0 is sharded on 'tp', dim 1 is replicated
        - PartitionSpec(('dp', 'tp'), None) means dim 0 is sharded on both 'dp' and 'tp'
        - PartitionSpec() means fully replicated
    """

    def __new__(cls, *args: str | tuple[str, ...] | None):
        return super().__new__(cls, args)

    def __repr__(self):
        pr = repr(tuple(self))[1:-1]
        if not self:
            return "PartitionSpec()"
        return f"PartitionSpec({pr})"


GlobalSpmdType: TypeAlias = "tuple[LocalSpmdType, PartitionSpec]"

# =============================================================================
# TensorSharding for Module Boundary Contracts
# =============================================================================

# Sharding for one tensor axis.
# * None: replicated;
# * str: the device mesh axis name, e.g., "fsdp" or "tp";
# * Sequence[str]: multiple device mesh axis names, e.g., ("fsdp", "cp").
DimSharding = None | str | Sequence[str]


def normalize_dim_sharding(dim_sharding: DimSharding) -> Sequence[str]:
    """Normalizes a sharding to a sequence of dim names."""
    if dim_sharding is None:
        return ()
    if isinstance(dim_sharding, str):
        return (dim_sharding,)
    return dim_sharding


class _PytreeTuple:
    """Tuple-like values that are treated as leaves of a PyTree.

    The purpose of this class is to allow ``TensorSharding``s to be treated as
    tree leaves (e.g., for compat with ``tree.map``).
    """

    def __init__(self, *values) -> None:
        self._values = tuple(values)

    def __repr__(self) -> str:
        pr = repr(self._values)[1:-1]
        return f"{type(self).__name__}({pr})"

    def __getitem__(self, i):
        return self._values[i]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self._values == other._values
        elif isinstance(other, tuple):
            return self._values == other
        return False

    def __hash__(self) -> int:
        return hash(self._values)

    def __add__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*self, *other)
        raise NotImplementedError(type(other))

    def __radd__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*other, *self)
        raise NotImplementedError(type(other))

    def index(self, value):
        return self._values.index(value)

    def count(self, value):
        return self._values.count(value)


class TensorSharding(_PytreeTuple):
    """A tuple of ``DimSharding``s describing the sharding for a Tensor.

    Each element maps one tensor dimension to zero, one, or multiple mesh axis
    names.  For example::

        TensorSharding("fsdp", "tp")   # dim 0 on fsdp, dim 1 on tp
        TensorSharding(None, "tp")     # dim 0 replicated, dim 1 on tp
        TensorSharding(("fsdp", "cp")) # dim 0 on both fsdp and cp

    Its length must be less than or equal to the number of Tensor axes.
    """

    def __init__(self, *dim_shardings: DimSharding) -> None:
        super().__init__(*dim_shardings)


class SpmdTypeError(RuntimeError):
    """Error raised for SPMD type mismatches.

    Inherits from RuntimeError (not TypeError) so that it is not swallowed by
    Python's binary-operator dispatch machinery when raised inside
    ``__torch_function__``.  Python interprets a TypeError from an operator
    dunder as "this type doesn't support the operation" and silently falls
    through to reflected operations, masking the real error message.
    """

    pass


def _canonicalize_shard(typ: PerMeshAxisSpmdType, ndim: int) -> PerMeshAxisSpmdType:
    """Resolve negative dims in Shard types. Returns typ unchanged if not Shard.

    Args:
        typ: The per-mesh-axis SPMD type, possibly a Shard with a negative dim.
        ndim: The number of dimensions of the tensor, used to resolve negative dims.
    """
    if isinstance(typ, Shard) and typ.dim < 0:
        return Shard(typ.dim % ndim)
    return typ


def format_axis(axis: DeviceMeshAxis) -> str:
    """Format a mesh axis for display in error messages.

    For string axes, returns the repr (e.g., ``'tp'``).
    For ProcessGroup axes, uses ``group_desc`` when available to produce a
    bare human-readable name (e.g., ``TP``) instead of the default opaque
    object repr.

    Args:
        axis: The mesh axis identifier, either a string name or a ProcessGroup.
    """
    if isinstance(axis, str):
        return repr(axis)
    # ProcessGroup - use group_desc for a readable name
    desc = getattr(axis, "group_desc", None)
    if desc and desc != "undefined":
        return desc
    return repr(axis)
