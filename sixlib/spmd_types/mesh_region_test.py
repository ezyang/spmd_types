"""Tests for explicit cross-mesh compatibility helpers."""

from __future__ import annotations

import unittest

from sixlib.spmd_types._mesh_axis import _reset, flatten_axes, MeshAxis
from sixlib.spmd_types._mesh_region import check_reinterpret_mesh_compatible
from sixlib.spmd_types.types import R, V


class TestFlattenAxes(unittest.TestCase):
    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_flatten_single_axis_is_identity(self) -> None:
        dp_cp = MeshAxis.of(4, 4)
        self.assertEqual(flatten_axes((dp_cp,)), dp_cp)

    def test_flatten_dp_cp(self) -> None:
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        dp_cp = MeshAxis.of(4, 4)
        self.assertEqual(flatten_axes((dp, cp)), dp_cp)

    def test_flatten_full_region(self) -> None:
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        tp = MeshAxis.of(4, 1)
        full = MeshAxis.of(16, 1)
        self.assertEqual(flatten_axes((dp, cp, tp)), full)

    def test_flatten_requires_orthogonal_axes(self) -> None:
        dp = MeshAxis.of(2, 8)
        dp_cp = MeshAxis.of(4, 4)
        with self.assertRaises(ValueError):
            flatten_axes((dp, dp_cp))


class TestReinterpretMeshCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_flatten_unflatten_is_compatible(self) -> None:
        dp_cp = MeshAxis.of(4, 4)
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        self.assertIsNone(check_reinterpret_mesh_compatible({dp: R, cp: R}, {dp_cp: R}))

    def test_parallel_folding_is_compatible(self) -> None:
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        tp = MeshAxis.of(4, 1)
        edp = MeshAxis.of(4, 4)
        ep = MeshAxis.of(2, 2)
        etp = MeshAxis.of(2, 1)
        self.assertIsNone(
            check_reinterpret_mesh_compatible(
                {dp: V, cp: V, tp: V},
                {edp: V, ep: V, etp: V},
            )
        )

    def test_mixed_group_types_are_incompatible(self) -> None:
        dp_cp = MeshAxis.of(4, 4)
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        err = check_reinterpret_mesh_compatible({dp: V, cp: R}, {dp_cp: V})
        self.assertIsNotNone(err)
        self.assertIn("Incompatible", err)

    def test_different_rank_patterns_are_incompatible(self) -> None:
        x = MeshAxis.of(4, 1)
        p = MeshAxis.of(4, 2)
        self.assertIsNotNone(check_reinterpret_mesh_compatible({x: R}, {p: R}))

    def test_empty_to_empty_is_compatible(self) -> None:
        self.assertIsNone(check_reinterpret_mesh_compatible({}, {}))


if __name__ == "__main__":
    unittest.main()
