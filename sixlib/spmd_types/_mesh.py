"""Context manager for setting the current mesh axes."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sixlib.spmd_types._mesh_axis import MeshAxis
from sixlib.spmd_types._state import _pop_mesh, _push_mesh
from sixlib.spmd_types.types import _check_orthogonality
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@contextmanager
def set_current_mesh(axes: frozenset[MeshAxis] | DeviceMesh | Sequence[ProcessGroup]):
    """Context manager that pushes a mesh onto the stack.

    Args:
        axes: The mesh to set. Accepts:
            - A frozenset of orthogonal MeshAxis objects.
            - A DeviceMesh whose named dimensions are converted to MeshAxis.
            - A sequence of ProcessGroup objects, each converted via MeshAxis.of().
    """
    resolved = _resolve_axes(axes)
    axes_list = list(resolved)
    _check_orthogonality(axes_list)
    _push_mesh(resolved)
    try:
        yield
    finally:
        _pop_mesh()


def _resolve_axes(
    axes: frozenset[MeshAxis] | DeviceMesh | Sequence[ProcessGroup],
) -> frozenset[MeshAxis]:
    """Normalize the various input forms to a frozenset of MeshAxis."""
    if isinstance(axes, frozenset):
        return axes

    if isinstance(axes, DeviceMesh):
        return _device_mesh_to_axes(axes)

    # Sequence of ProcessGroup
    return frozenset(MeshAxis.of(pg) for pg in axes)


def _device_mesh_to_axes(mesh: DeviceMesh) -> frozenset[MeshAxis]:
    """Convert a DeviceMesh to a frozenset of MeshAxis objects.

    Each named dimension of the mesh becomes a MeshAxis via
    ``MeshAxis.of(mesh.get_group(name))``.
    """
    if not mesh.mesh_dim_names:
        raise ValueError(
            "DeviceMesh must have mesh_dim_names set to be used with "
            "set_current_mesh. Use init_device_mesh(..., mesh_dim_names=...) "
            "or pass a frozenset of MeshAxis objects instead."
        )
    return frozenset(MeshAxis.of(mesh.get_group(name)) for name in mesh.mesh_dim_names)
