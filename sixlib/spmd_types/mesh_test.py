"""
Tests for mesh stack: set_current_mesh / current_mesh.

Covers: _mesh.py, _state.py (mesh stack), _checker.py (infer_output_type integration).
"""

import unittest

import torch
from sixlib.spmd_types import (
    current_mesh,
    MeshAxis,
    R,
    set_current_mesh,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_axis_local_type,
    typecheck,
)
from sixlib.spmd_types._state import _clear_mesh_stack
from sixlib.spmd_types._test_utils import LocalTensorTestCase, SpmdTypeCheckedTestCase
from sixlib.spmd_types.types import SpmdTypeError
from torch.distributed.device_mesh import init_device_mesh


class TestMeshStackState(SpmdTypeCheckedTestCase):
    """Test push/pop/current_mesh state management."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")
        cls.tp_axis = MeshAxis.of(cls.tp)
        cls.dp_axis = MeshAxis.of(cls.dp)

    def setUp(self):
        _clear_mesh_stack()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        _clear_mesh_stack()

    def test_no_mesh_returns_none(self):
        """current_mesh() returns None when no mesh is set."""
        self.assertIsNone(current_mesh())

    def test_set_current_mesh_basic(self):
        """set_current_mesh pushes mesh, pops on exit."""
        mesh = frozenset({self.tp_axis, self.dp_axis})
        with set_current_mesh(mesh):
            self.assertEqual(current_mesh(), mesh)
        self.assertIsNone(current_mesh())

    def test_set_current_mesh_from_device_mesh(self):
        """set_current_mesh accepts a DeviceMesh directly."""
        device_mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        expected = frozenset({self.tp_axis, self.dp_axis})
        with set_current_mesh(device_mesh):
            self.assertEqual(current_mesh(), expected)
        self.assertIsNone(current_mesh())

    def test_set_current_mesh_from_process_groups(self):
        """set_current_mesh accepts a sequence of ProcessGroups."""
        expected = frozenset({self.tp_axis, self.dp_axis})
        with set_current_mesh([self.tp, self.dp]):
            self.assertEqual(current_mesh(), expected)
        self.assertIsNone(current_mesh())

    def test_nesting(self):
        """Inner set_current_mesh overrides outer, restores on exit."""
        outer = frozenset({self.tp_axis})
        inner = frozenset({self.dp_axis})
        with set_current_mesh(outer):
            self.assertEqual(current_mesh(), outer)
            with set_current_mesh(inner):
                self.assertEqual(current_mesh(), inner)
            self.assertEqual(current_mesh(), outer)
        self.assertIsNone(current_mesh())

    def test_orthogonality_check(self):
        """Non-orthogonal axes in set_current_mesh raise SpmdTypeError."""
        from sixlib.spmd_types._mesh_axis import flatten_axes

        # dp_tp covers all ranks, so it overlaps with both dp and tp
        dp_tp = flatten_axes((self.dp_axis, self.tp_axis))
        with self.assertRaises(SpmdTypeError):
            with set_current_mesh(frozenset({self.tp_axis, dp_tp})):
                pass

    def test_pops_on_exception(self):
        """Mesh is popped even when the body raises."""
        mesh = frozenset({self.tp_axis})

        class _TestError(Exception):
            pass

        with self.assertRaises(_TestError):
            with set_current_mesh(mesh):
                self.assertEqual(current_mesh(), mesh)
                raise _TestError("boom")
        self.assertIsNone(current_mesh())


class TestMeshStrictMode(SpmdTypeCheckedTestCase):
    """Test strict mode + current mesh interaction.

    SpmdTypeCheckedTestCase enters typecheck() (default strict mode) and
    sets up a fake process group at self.pg.
    """

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")
        cls.tp_axis = MeshAxis.of(cls.tp)
        cls.dp_axis = MeshAxis.of(cls.dp)

    def test_strict_missing_mesh_axis_raises(self):
        """Strict mode: tensor missing a mesh axis from current_mesh raises."""
        mesh = frozenset({self.tp_axis, self.dp_axis})
        # x is annotated on tp only, not dp
        x = self._generate_inputs((4,), self.tp, R)
        with set_current_mesh(mesh):
            with self.assertRaises(SpmdTypeError):
                torch.add(x, x)

    def test_strict_all_mesh_axes_present_passes(self):
        """Strict mode: tensor annotated on all mesh axes passes."""
        mesh = frozenset({self.tp_axis, self.dp_axis})
        x = self._generate_inputs((4,), self.tp, R)
        assert_type(x, {self.dp: R})
        with set_current_mesh(mesh):
            result = torch.add(x, x)
            self.assertIs(get_axis_local_type(result, self.tp), R)
            self.assertIs(get_axis_local_type(result, self.dp), R)

    def test_no_mesh_unchanged_behavior(self):
        """Without current mesh, strict mode works as before."""
        x = self._generate_inputs((4,), self.tp, R)
        result = torch.add(x, x)
        self.assertIs(get_axis_local_type(result, self.tp), R)

    def test_assert_type_does_not_enforce_mesh(self):
        """assert_type does not enforce the current mesh -- ops do."""
        mesh = frozenset({self.tp_axis, self.dp_axis})
        with set_current_mesh(mesh):
            x = self.rank_map(lambda r: torch.randn(4))
            # Partial annotation succeeds -- assert_type is just annotation
            assert_type(x, {self.tp: R})
            self.assertIs(get_axis_local_type(x, self.tp), R)
            # But using the tensor in an op fails due to missing dp
            with self.assertRaises(SpmdTypeError):
                torch.add(x, x)


class TestMeshPermissiveMode(LocalTensorTestCase):
    """Test permissive mode + current mesh (missing axes are skipped)."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")
        cls.tp_axis = MeshAxis.of(cls.tp)
        cls.dp_axis = MeshAxis.of(cls.dp)

    def test_permissive_missing_mesh_axis_no_error(self):
        """Permissive mode: missing mesh axis is skipped, no error."""
        mesh = frozenset({self.tp_axis, self.dp_axis})
        with typecheck(strict_mode="permissive"):
            # x is annotated on tp only, not dp
            x = self._generate_inputs((4,), self.tp, R)
            with set_current_mesh(mesh):
                result = torch.add(x, x)
                self.assertIs(get_axis_local_type(result, self.tp), R)


if __name__ == "__main__":
    unittest.main()
