"""Helpers for explicit cross-mesh compatibility on local SPMD types.

This module supports the two mesh-compatibility cases used by explicit mesh
reinterpretation:

* a flattened axis directly matching a tuple of orthogonal axes
* two tuples of orthogonal axes that flatten to the same larger region

Compatibility is checked by sorting axes by stride descending (outermost
first) on each side and walking both lists with two pointers.  Group
boundaries appear when the cumulative sizes align; at each boundary the
flattened regions and uniform local types must match.
"""

from __future__ import annotations

from sixlib.spmd_types._mesh_axis import flatten_axes, MeshAxis
from sixlib.spmd_types.types import LocalSpmdType, PerMeshAxisLocalSpmdType


def _max_stride(axis: MeshAxis) -> int:
    """Return the maximum stride across all atoms of an axis."""
    return max(d for _, d in axis.layout.sizes_and_strides)


def check_reinterpret_mesh_compatible(
    src: LocalSpmdType,
    dst: LocalSpmdType,
) -> str | None:
    """Check whether ``src`` can be explicitly reinterpreted as ``dst``.

    Returns None if compatible, or an error message string if not.

    Exactly matching axes must carry the same local type.  Remaining axes are
    sorted by stride descending and walked with two pointers; group boundaries
    appear when cumulative sizes align.  At each boundary the flattened regions
    and uniform local types must match.
    """
    shared_axes = src.keys() & dst.keys()
    for axis in shared_axes:
        if src[axis] is not dst[axis]:
            return (
                f"Shared mesh axis {axis} changes local type from {src[axis]} to "
                f"{dst[axis]} under reinterpret_mesh."
            )

    src_remaining = {axis: typ for axis, typ in src.items() if axis not in shared_axes}
    dst_remaining = {axis: typ for axis, typ in dst.items() if axis not in shared_axes}
    if not src_remaining and not dst_remaining:
        return None
    if not src_remaining or not dst_remaining:
        return (
            f"Cross-mesh reinterpretation is incomplete: {src_remaining or src} vs "
            f"{dst_remaining or dst}."
        )

    # Sort axes by stride descending (outermost first).
    src_sorted = sorted(
        src_remaining.items(), key=lambda item: _max_stride(item[0]), reverse=True
    )
    dst_sorted = sorted(
        dst_remaining.items(), key=lambda item: _max_stride(item[0]), reverse=True
    )

    return _walk_groups(src_sorted, dst_sorted, src, dst)


def _walk_groups(
    src_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType]],
    dst_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType]],
    src: LocalSpmdType,
    dst: LocalSpmdType,
) -> str | None:
    """Two-pointer walk over stride-sorted axes, checking groups at each boundary."""
    i = j = 0
    src_cum = dst_cum = 1
    src_group: list[MeshAxis] = []
    dst_group: list[MeshAxis] = []
    src_group_type: PerMeshAxisLocalSpmdType | None = None
    dst_group_type: PerMeshAxisLocalSpmdType | None = None

    while i < len(src_sorted) or j < len(dst_sorted):
        # Advance whichever side has the smaller cumulative size.
        if src_cum <= dst_cum and i < len(src_sorted):
            axis, typ = src_sorted[i]
            src_cum *= axis.size()
            src_group.append(axis)
            if src_group_type is None:
                src_group_type = typ
            elif src_group_type is not typ:
                return (
                    f"Incompatible cross-mesh local types: {src} vs {dst}. "
                    f"Source group {src_group} has mixed local types."
                )
            i += 1
        elif j < len(dst_sorted):
            axis, typ = dst_sorted[j]
            dst_cum *= axis.size()
            dst_group.append(axis)
            if dst_group_type is None:
                dst_group_type = typ
            elif dst_group_type is not typ:
                return (
                    f"Incompatible cross-mesh local types: {src} vs {dst}. "
                    f"Destination group {dst_group} has mixed local types."
                )
            j += 1
        else:
            # src side exhausted but dst_cum < src_cum -- sizes don't align.
            break

        # Check for group boundary.
        if src_cum == dst_cum:
            assert src_group_type is not None and dst_group_type is not None
            if src_group_type is not dst_group_type:
                return (
                    f"Incompatible cross-mesh local types: {src} vs {dst}. "
                    f"Source group {src_group} has type {src_group_type} but "
                    f"destination group {dst_group} has type {dst_group_type}."
                )
            if flatten_axes(tuple(src_group)) != flatten_axes(tuple(dst_group)):
                return (
                    f"Incompatible cross-mesh local types: {src} vs {dst}. "
                    f"Source group {src_group} and destination group {dst_group} "
                    "flatten to different rank regions."
                )
            # Reset for next group.
            src_cum = dst_cum = 1
            src_group = []
            dst_group = []
            src_group_type = dst_group_type = None

    if src_cum != 1 or dst_cum != 1:
        return (
            f"Incompatible cross-mesh local types: {src} vs {dst}. The non-shared "
            "axes cannot be grouped into equal-rank regions with matching local types."
        )

    return None
