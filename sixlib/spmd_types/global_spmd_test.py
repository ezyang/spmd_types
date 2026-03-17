"""
Tests for global SPMD shard propagation: propagating S(i) through torch ops.

In global SPMD, the per-axis types are R, I, P, and S(i) (no V). S(i)
propagation reuses DTensor's sharding propagation to determine how shard
dimensions flow through ops, while local SPMD inference handles R/I/P.

Decision tree per mesh axis (when S is present):
1. Replace S->V, run local SPMD as compatibility check.
2. If local SPMD rejects -> propagate error.
3. If local SPMD accepts -> run DTensor shard propagation for S(i) output.

When no S is present: local SPMD only.

When DTensor says output is Partial but local SPMD (with S->V) says Varying,
the operation is rejected: the implicit partial reduction must be handled
explicitly (e.g. all_gather before reducing, or use local_map).
"""

import torch
from sixlib.spmd_types import (
    MeshAxis,
    PartitionSpec,
    R,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_local_type,
    get_partition_spec,
)
from sixlib.spmd_types._test_utils import SpmdTypeCheckedTestCase
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    normalize_axis,
    partition_spec_to_shard_types,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    SpmdTypeError,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import run_tests


def _get_global_type(tensor: torch.Tensor) -> PerMeshAxisSpmdTypes:
    """Reconstruct S(i) from local type + PartitionSpec (test helper)."""
    typ = get_local_type(tensor).copy()
    spec = get_partition_spec(tensor)
    if spec:
        for axis, shard in partition_spec_to_shard_types(spec).items():
            typ[normalize_axis(axis)] = shard
    return typ


def _get_axis_type(tensor: torch.Tensor, axis: DeviceMeshAxis) -> PerMeshAxisSpmdType:
    """Get the full SPMD type for a specific axis, including S(i)."""
    return _get_global_type(tensor).get(normalize_axis(axis), V)


# =============================================================================
# S(i) storage
# =============================================================================


class TestShardTypeStorage(SpmdTypeCheckedTestCase):
    """Test that S(i) can be stored on and read from tensors."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = init_device_mesh("cpu", (cls.WORLD_SIZE,), mesh_dim_names=("tp",))
        cls.tp = cls.mesh.get_group("tp")
        cls.tp_key = normalize_axis(cls.tp)
        # dp/ep: standalone MeshAxis objects with distinct strides.
        cls.dp = MeshAxis.of(cls.WORLD_SIZE, stride=cls.WORLD_SIZE)
        cls.dp_key = cls.dp
        cls.ep = MeshAxis.of(cls.WORLD_SIZE, stride=cls.WORLD_SIZE**2)
        cls.ep_key = cls.ep

    def test_assert_type_stores_shard(self):
        """assert_type with S(i) stores PartitionSpec on tensor."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        self.assertEqual(_get_axis_type(x, self.tp), S(0))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp_key, None))

    def test_assert_type_stores_shard_with_other_axes(self):
        """S(i) on one axis, R on another."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0), self.tp: R})
        self.assertEqual(_get_axis_type(x, self.dp), S(0))
        self.assertEqual(_get_axis_type(x, self.tp), R)

    def test_assert_type_shard_check_matches(self):
        """Re-asserting the same S(i) should succeed."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.tp: S(0)})

    def test_assert_type_shard_check_mismatch_raises(self):
        """Re-asserting conflicting S(i) should raise."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(1)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: R})

    def test_check_v_then_r(self):
        """Asserting V then R on same axis."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: V})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: R})

    def test_check_r_then_v(self):
        """Asserting R then V on same axis."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: R})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: V})

    def test_get_local_type_returns_v_for_sharded(self):
        """get_local_type returns V for sharded axes."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        local = get_local_type(x)
        self.assertIs(local[self.tp_key], V)
        # get_global_type reconstructs S from PartitionSpec.
        self.assertEqual(_get_global_type(x)[self.tp_key], S(0))

    def test_get_local_type_preserves_non_shard_types(self):
        """get_local_type returns R/I/P normally."""
        x = self._generate_inputs((4, 3), self.pg, R)
        self.assertIs(get_local_type(x)[normalize_axis(self.pg)], R)

    def test_get_global_type_reconstructs_s(self):
        """get_global_type reconstructs S from PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        gtype = _get_global_type(x)
        self.assertEqual(gtype[self.tp_key], S(0))
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec(self.tp_key, None))

    def test_negative_dim_shard(self):
        """S(-1) resolves to S(ndim-1)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(-1)})
        self.assertEqual(_get_axis_type(x, self.tp), S(1))
        self.assertEqual(get_partition_spec(x), PartitionSpec(None, self.tp_key))

    def test_recheck_v_then_s(self):
        """Asserting V then S(i) on same axis refines with PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: V})
        assert_type(x, {self.tp: S(0)})
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp_key, None))

    def test_recheck_s_then_v(self):
        """Asserting S(i) then V keeps existing PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.tp: V})  # V is less specific, doesn't overwrite
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp_key, None))

    def test_conflicting_s_rejected(self):
        """Same axis, different dim -> SpmdTypeError."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(1)})

    def test_multi_axis_same_dim_with_partition_spec(self):
        """Multi-axis on same dim requires explicit PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp_key, self.tp_key), None),
        )
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec((self.dp_key, self.tp_key), None))

    def test_merge_different_axes_different_dims(self):
        """S(i) on tp then S(j) on dp merges into combined PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.dp: S(1)})
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp_key, self.dp_key))

    def test_recheck_spec_refine_none_to_sharded(self):
        """PartitionSpec(dp, None) then PartitionSpec(dp, tp) refines dim 1."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp_key, None))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec(self.dp_key, self.tp_key),
        )
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp_key, self.tp_key))

    def test_recheck_spec_single_to_multi_axis_rejected(self):
        """PartitionSpec(dp) then PartitionSpec((dp, tp)) conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp_key, None))
        with self.assertRaises(SpmdTypeError):
            assert_type(
                x,
                {self.dp: V, self.tp: V},
                PartitionSpec((self.dp_key, self.tp_key), None),
            )

    def test_multi_axis_same_dim_without_spec_rejected(self):
        """{tp: S(0)} then {dp: S(0)}: error without explicit PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: S(0)})

    def test_recheck_s_then_consistent_spec(self):
        """{dp: S(0)} then PartitionSpec(dp, None): OK (consistent)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp_key, None))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp_key, None))

    def test_recheck_s_then_spec_adds_new_axis(self):
        """{dp: S(0)} then PartitionSpec(dp, tp): OK, adds tp on dim 1."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec(self.dp_key, self.tp_key),
        )
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp_key, self.tp_key))

    def test_partition_spec_ordering_matters(self):
        """PartitionSpec ordering matters: (tp, dp) != (dp, tp)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.tp: V, self.dp: V},
            PartitionSpec((self.tp_key, self.dp_key), None),
        )
        y = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            y,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp_key, self.tp_key), None),
        )
        self.assertNotEqual(get_partition_spec(x), get_partition_spec(y))

    def test_recheck_s_to_multi_axis_rejected(self):
        """S(0) on dp then PartitionSpec((dp, tp)) on same dim conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(
                x,
                {self.dp: V, self.tp: V},
                PartitionSpec((self.dp_key, self.tp_key), None),
            )

    def test_recheck_multi_axis_to_s_rejected(self):
        """PartitionSpec((dp, tp)) then S(0) on dp conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp_key, self.tp_key), None),
        )
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: S(0)})

    def test_explicit_v_is_stored(self):
        """assert_type with explicit V stores V in the local type dict."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.ep: V, self.tp: R})
        local = get_local_type(x)
        self.assertIs(local[self.ep_key], V)
        self.assertIs(local[self.tp_key], R)


if __name__ == "__main__":
    run_tests()
