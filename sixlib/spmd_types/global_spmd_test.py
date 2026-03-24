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

import unittest

import torch
import torch.distributed as dist
from sixlib.spmd_types import (
    MeshAxis,
    P,
    PartitionSpec,
    R,
    S,
    V,
)
from sixlib.spmd_types._checker import (
    _collect_shard_axes,
    _infer_global_output_type,
    _set_partition_spec,
    _validate_and_update_local_global_correspondence,
    assert_type,
    dtensor_placement_to_spmd_type,
    get_local_type,
    get_partition_spec,
    typecheck,
)
from sixlib.spmd_types._mesh_axis import _reset
from sixlib.spmd_types._test_utils import SpmdTypeCheckedTestCase
from sixlib.spmd_types.types import (
    DeviceMeshAxis,
    normalize_axis,
    partition_spec_to_shard_types,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    SpmdTypeError,
    to_local_type,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore


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


# =============================================================================
# _collect_shard_axes
# =============================================================================


class TestCollectShardAxes(unittest.TestCase):
    """Test _collect_shard_axes extracts and orders shard axes from PartitionSpecs."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=64, store=store)
        cls.mesh = init_device_mesh("cpu", (4, 4, 4), mesh_dim_names=("ep", "dp", "tp"))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        self.tp = normalize_axis(self.mesh.get_group("tp"))
        self.dp = normalize_axis(self.mesh.get_group("dp"))
        self.ep = normalize_axis(self.mesh.get_group("ep"))

    def test_no_partition_spec_returns_empty(self):
        """Tensor without PartitionSpec contributes no axes."""
        result, edges = _collect_shard_axes([None], {self.tp})
        self.assertEqual(result, [])
        self.assertEqual(edges, {})

    def test_multi_axis_tuple_ordering(self):
        """Tuple entry (tp, dp) yields tp before dp."""
        specs = [
            PartitionSpec(None, (self.tp, self.dp), None),
            PartitionSpec((self.tp, self.dp), None, None),
            PartitionSpec((self.dp, self.ep), None, None),
        ]
        result, edges = _collect_shard_axes(specs, {self.tp, self.dp, self.ep})
        self.assertEqual(result, [self.tp, self.dp, self.ep])
        # Verify edges: tp->dp, dp->ep, tp->ep
        self.assertIn(self.dp, edges[self.tp])
        self.assertIn(self.ep, edges[self.dp])

    def test_ordering_conflict_raises(self):
        """Conflicting orderings across inputs raise SpmdTypeError."""
        specs = [
            PartitionSpec((self.tp, self.dp), None, None),
            PartitionSpec((self.dp, self.tp), None, None),
        ]
        with self.assertRaises(SpmdTypeError):
            _collect_shard_axes(specs, {self.tp, self.dp})


# =============================================================================
# DTensor shard propagator and _infer_global_output_type
# =============================================================================


class TestInferGlobalOutputType(unittest.TestCase):
    """Test _infer_global_output_type with plain tensors.

    Uses plain tensors with SPMD type annotations (no LocalTensorMode)
    since propagation only needs tensor shapes, dtypes, and type metadata.
    """

    WORLD_SIZE = 3

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.mesh = init_device_mesh("cpu", (cls.WORLD_SIZE,), mesh_dim_names=("tp",))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        self.tp = normalize_axis(self.mesh.get_group("tp"))

    def _make_typed(self, shape, typ):
        """Create a plain tensor with SPMD type and optional PartitionSpec."""
        x = torch.randn(shape)
        local_typ = V if isinstance(typ, S) else typ
        with typecheck(strict_mode="permissive"):
            assert_type(x, {self.tp: local_typ})
        if isinstance(typ, S):
            spec = [self.tp if d == typ.dim else None for d in range(len(shape))]
            _set_partition_spec(x, PartitionSpec(*spec))
        return x

    def test_add_s0_s0(self):
        """add(S(0), S(0)) -> output PartitionSpec has S(0)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(0))
        result = torch.add(x, y)
        pls, specs = _infer_global_output_type(
            func=torch.add,
            args=(x, y),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(tp, None))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_add_s0_s0_out_mismatch(self):
        """add(S(0), S(0), out=out) -> should fail due to output PartitionSpec
        mismatch."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(0))
        out = self._make_typed((8, 3), S(1))
        result = torch.add(x, y)
        with self.assertRaises(SpmdTypeError):
            _infer_global_output_type(
                func=torch.add,
                args=(x, y),
                kwargs={"out": out},
                global_shard_axes=[tp],
                flat_results=[result],
            )

    def test_add_s0_s1(self):
        """add(S(0), S(1)) -> should fail for DTensor propagation due to
        redistribution needed."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(1))
        result = torch.add(x, y)
        with self.assertRaises(SpmdTypeError):
            _infer_global_output_type(
                func=torch.add,
                args=(x, y),
                kwargs=None,
                global_shard_axes=[tp],
                flat_results=[result],
            )

    def test_mm_s0_r(self):
        """matmul(S(0), R) -> output PartitionSpec has S(0)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        w = self._make_typed((3, 5), R)
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(tp, None))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_mm_r_s1(self):
        """matmul(R, S(1)) -> output PartitionSpec has S(1)."""
        tp = self.tp
        x = self._make_typed((8, 3), R)
        w = self._make_typed((3, 5), S(1))
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(None, tp))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_mm_s1_s0_gives_partial(self):
        """matmul(S(1), S(0)) -> Partial: accepted by correspondence check."""
        tp = self.tp
        x = self._make_typed((8, 3), S(1))
        w = self._make_typed((3, 5), S(0))
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        # DTensor produces Partial; local SPMD produces V.
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        out = _validate_and_update_local_global_correspondence(V, global_results, tp)
        self.assertEqual(out, [P])

    def test_transpose_s0_gives_s1(self):
        """transpose(S(0), 0, 1) -> output PartitionSpec has S(1)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        result = torch.transpose(x, 0, 1)
        pls, specs = _infer_global_output_type(
            func=torch.transpose,
            args=(x, 0, 1),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(None, tp))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)


if __name__ == "__main__":
    run_tests()
