"""
Tests for global SPMD shard propagation: propagating S(i) through torch ops.

In global SPMD, the per-axis types are R, I, P, and S(i) (no V). S(i)
propagation reuses DTensor's sharding propagation to determine how shard
dimensions flow through ops, while local SPMD inference handles R/I/P.

Decision tree per mesh axis (when S is present):
1. Replace S→V, run local SPMD as compatibility check.
2. If local SPMD rejects → propagate error.
3. If local SPMD accepts → run DTensor shard propagation for S(i) output.

When no S is present: local SPMD only.

When DTensor says output is Partial but local SPMD (with S→V) says Varying,
the operation is rejected: the implicit partial reduction must be handled
explicitly (e.g. all_gather before reducing, or use local_map).
"""

import torch
from sixlib.spmd_types import (
    I,
    P,
    R,
    S,
    V,
    all_reduce,
    local_map,
    redistribute,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_local_type,
    has_local_type,
    SpmdTypeMode,
)
from sixlib.spmd_types._test_utils import LocalTensorTestCase
from sixlib.spmd_types.types import PerMeshAxisSpmdType, Shard, SpmdTypeError
from torch.distributed._local_tensor import LocalTensorMode
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


def _type_name(t):
    """Short name for SPMD types, used in parametrized test names."""
    return t.name if hasattr(t, "name") else str(t)


def _get_axis_type(tensor: torch.Tensor, axis: str) -> PerMeshAxisSpmdType:
    """Get the full SPMD type for a specific axis, including S(i).

    Unlike get_axis_local_type (which decays S to V), this returns the stored
    type which may include S(i) in global SPMD.
    """
    return get_local_type(tensor).get(axis, V)


class GlobalSpmdTestCase(LocalTensorTestCase):
    """Base test class for global SPMD tests.

    Like LocalTensorTestCase, but uses SpmdTypeMode(global_mode=True) to enable
    S(i) storage and DTensor-based shard propagation.
    """

    def setUp(self):
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self.spmd_mode = SpmdTypeMode(global_mode=True)
        self.spmd_mode.__enter__()

    def _make_input(self, shape, axis, typ):
        if typ is R or typ is I:
            base = torch.randn(shape)
            result = self.mode.rank_map(lambda r: base.clone())
        else:
            result = self.mode.rank_map(lambda r: torch.randn(shape) + r)
        assert_type(result, {axis: typ})
        return result

    def _make_multi_axis_input(self, shape, types):
        has_shard_or_varying = any(
            isinstance(t, Shard) or t is V or t is P for t in types.values()
        )
        if not has_shard_or_varying:
            base = torch.randn(shape)
            result = self.mode.rank_map(lambda r: base.clone())
        else:
            result = self.mode.rank_map(lambda r: torch.randn(shape) + r)
        assert_type(result, types)
        return result


# =============================================================================
# S(i) storage
# =============================================================================


class TestShardTypeStorage(GlobalSpmdTestCase):
    """Test that S(i) can be stored on and read from tensors."""

    def test_assert_type_reject_varying(self):
        """V is not a valid type in global SPMD."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": V})

    def test_assert_type_stores_shard(self):
        """assert_type({'tp': S(0)}) should store S(0), not decay to V."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        self.assertEqual(_get_axis_type(x, "tp"), S(0))

    def test_assert_type_stores_shard_with_other_axes(self):
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"dp": S(0), "tp": R})
        self.assertEqual(_get_axis_type(x, "dp"), S(0))
        self.assertEqual(_get_axis_type(x, "tp"), R)

    def test_assert_type_shard_check_matches(self):
        """Re-asserting the same S(i) type should succeed."""
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        assert_type(x, {"tp": S(0)})

    def test_assert_type_shard_check_mismatch_raises(self):
        x = self.mode.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {"tp": S(0)})
        with self.assertRaises(AssertionError):
            assert_type(x, {"tp": S(1)})
        with self.assertRaises(AssertionError):
            assert_type(x, {"tp": R})


# =============================================================================
# S(i) propagation through pointwise ops
# =============================================================================


class TestGlobalSpmdPointwise(GlobalSpmdTestCase):
    """Test S(i) propagation through pointwise/elementwise ops."""

    @parametrize("op", [
        torch.neg, torch.abs, torch.exp, torch.tanh,
        torch.sigmoid, torch.relu,
    ], name_fn=lambda op: op.__name__)
    def test_unary_s0(self, op):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(op(x), "tp"), S(0))

    @parametrize("op", [torch.sqrt, torch.rsqrt, torch.reciprocal],
                 name_fn=lambda op: op.__name__)
    def test_positive_unary_s0(self, op):
        """Ops requiring positive input: apply abs+1 first."""
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(op(torch.abs(x) + 1), "tp"), S(0))

    def test_clone_s0(self):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(x.clone(), "tp"), S(0))

    @parametrize("op", [torch.add, torch.sub, torch.mul],
                 name_fn=lambda op: op.__name__)
    def test_binary_s0_s0(self, op):
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(op(x, y), "tp"), S(0))

    @parametrize("op", [torch.add, torch.mul, torch.div],
                 name_fn=lambda op: op.__name__)
    def test_s0_r_rejected(self, op):
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            op(x, y)

    def test_r_s0_rejected(self):
        """R + S(0) rejected: order doesn't matter."""
        x = self._make_input((4, 3), "tp", R)
        y = self._make_input((4, 3), "tp", S(0))
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_s0_r_fix_with_redistribute(self):
        """S(0) + R rejected, but redistribute R -> S(0) fixes it."""
        x = self._make_input((4, 3), "tp", S(0))
        # R tensor has global shape (12, 3) to match S(0) global shape
        y = self._make_input((12, 3), "tp", R)
        with self.assertRaises(SpmdTypeError):
            x + y
        y_s = redistribute(y, "tp", src=R, dst=S(0))
        result = x + y_s
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    @parametrize("op,scalar", [
        (torch.add, 1), (torch.mul, 2.0),
        (torch.sub, 0.5), (torch.div, 3.0),
    ], name_fn=lambda op, scalar: op.__name__)
    def test_scalar(self, op, scalar):
        x = self._make_input((4, 3), "tp", S(0))
        self.assertEqual(_get_axis_type(op(x, scalar), "tp"), S(0))


instantiate_parametrized_tests(TestGlobalSpmdPointwise)


# =============================================================================
# S(i) propagation through unary shape-changing ops
# =============================================================================


class TestGlobalSpmdUnaryOps(GlobalSpmdTestCase):
    """Test S(i) propagation through transpose and reduction ops."""

    @parametrize("input_type,expected", [
        (S(0), S(1)),
        (S(1), S(0)),
    ], name_fn=lambda input_type, expected: f"{_type_name(input_type)}_to_{_type_name(expected)}")
    def test_transpose(self, input_type, expected):
        x = self._make_input((6, 4), "tp", input_type)
        self.assertEqual(_get_axis_type(torch.t(x), "tp"), expected)

    def test_sum_s0_rejected(self):
        """Reducing a sharded tensor implicitly produces Partial — rejected."""
        x = self._make_input((6, 4), "tp", S(0))
        with self.assertRaises(SpmdTypeError):
            torch.sum(x)

    def test_transpose_then_mm(self):
        """t(S(0)) -> S(1), mm(R, S(1)) -> S(1)."""
        x = self._make_input((4, 6), "tp", R)
        w = self._make_input((3, 6), "tp", S(0))
        w_t = torch.t(w)
        self.assertEqual(_get_axis_type(w_t, "tp"), S(1))
        result = torch.mm(x, w_t)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))


instantiate_parametrized_tests(TestGlobalSpmdUnaryOps)


# =============================================================================
# S(i) propagation through matmul ops
# =============================================================================


class TestGlobalSpmdMatmul(GlobalSpmdTestCase):
    """Test S(i) propagation through matmul ops."""

    @parametrize("op,x_shape,w_shape,x_type,w_type,expected", [
        (torch.mm, (4, 3), (3, 5), S(0), R, S(0)),
        (torch.mm, (4, 3), (3, 5), R, S(1), S(1)),
        (torch.mm, (4, 3), (3, 5), R, R, R),
        (torch.matmul, (4, 3), (3, 5), S(0), R, S(0)),
        (torch.bmm, (2, 6, 3), (2, 3, 4), S(1), R, S(1)),
        (torch.bmm, (2, 4, 3), (2, 3, 6), R, S(2), S(2)),
    ], name_fn=lambda op, x_shape, w_shape, x_type, w_type, expected: (
        f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
    ))
    def test_mm(self, op, x_shape, w_shape, x_type, w_type, expected):
        x = self._make_input(x_shape, "tp", x_type)
        w = self._make_input(w_shape, "tp", w_type)
        self.assertEqual(_get_axis_type(op(x, w), "tp"), expected)

    def test_matmul_operator_s0_r(self):
        """x @ w with S(0)@R -> S(0) via __matmul__."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        self.assertEqual(_get_axis_type(x @ w, "tp"), S(0))


instantiate_parametrized_tests(TestGlobalSpmdMatmul)


# =============================================================================
# Implicit Partial rejected (local SPMD says V but DTensor says P)
# =============================================================================


class TestGlobalSpmdRejection(GlobalSpmdTestCase):
    """Test cases that should be rejected in global SPMD."""

    @parametrize("op,x_shape,w_shape,x_type,w_type", [
        # S + P rejection
        (torch.add, (4, 3), (4, 3), S(0), P),
        (torch.mul, (4, 3), (4, 3), S(0), P),
        (torch.mm, (4, 3), (3, 5), S(0), P),
        # Incompatible shard dims
        (torch.add, (4, 3), (4, 3), S(0), S(1)),
        # mm: contracted sharded dim produces implicit Partial
        (torch.mm, (4, 3), (3, 5), S(1), S(0)),
        # mm: weight K dim sharded, no feasible strategy
        (torch.mm, (4, 3), (3, 5), R, S(0)),
    ], name_fn=lambda op, x_shape, w_shape, x_type, w_type: (
        f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
    ))
    def test_rejected(self, op, x_shape, w_shape, x_type, w_type):
        x = self._make_input(x_shape, "tp", x_type)
        w = self._make_input(w_shape, "tp", w_type)
        with self.assertRaises(SpmdTypeError):
            op(x, w)

    def test_add_s_p_has_fix_suggestion(self):
        """S+P rejection goes through local SPMD and gets fix suggestions."""
        x = self._make_input((4, 3), "tp", S(0))
        y = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError) as ctx:
            x + y
        self.assertIn("all_reduce", str(ctx.exception))


instantiate_parametrized_tests(TestGlobalSpmdRejection)


# =============================================================================
# I fallback to local SPMD inference
# =============================================================================


class TestGlobalSpmdInvariant(GlobalSpmdTestCase):
    """Test I type: I+I passes via local SPMD, I mixed with S/R/P is rejected."""

    @parametrize("op,x_shape,w_shape", [
        (torch.add, (4, 3), (4, 3)),
        (torch.mul, (4, 3), (4, 3)),
        (torch.mm, (4, 3), (3, 5)),
    ], name_fn=lambda op, x_shape, w_shape: op.__name__)
    def test_i_i_ok(self, op, x_shape, w_shape):
        x = self._make_input(x_shape, "tp", I)
        w = self._make_input(w_shape, "tp", I)
        self.assertEqual(_get_axis_type(op(x, w), "tp"), I)

    @parametrize("op,x_shape,w_shape,x_type,w_type", [
        (torch.add, (4, 3), (4, 3), I, R),
        (torch.mul, (4, 3), (4, 3), I, R),
        (torch.add, (4, 3), (4, 3), I, P),
        (torch.mm, (4, 3), (3, 5), I, R),
        (torch.mm, (4, 3), (3, 5), S(0), I),
        (torch.mm, (4, 6), (6, 5), I, S(1)),
        (torch.add, (4, 3), (4, 3), S(0), I),
    ], name_fn=lambda op, x_shape, w_shape, x_type, w_type: (
        f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
    ))
    def test_i_mixed_rejected(self, op, x_shape, w_shape, x_type, w_type):
        x = self._make_input(x_shape, "tp", x_type)
        w = self._make_input(w_shape, "tp", w_type)
        with self.assertRaises(SpmdTypeError):
            op(x, w)


instantiate_parametrized_tests(TestGlobalSpmdInvariant)


# =============================================================================
# R/P combinations without shard axes
# =============================================================================


class TestGlobalSpmdNoShardAxes(GlobalSpmdTestCase):
    """Test R and P type interactions in global SPMD (no S involved)."""

    @parametrize("op,x_shape,w_shape,x_type,w_type,expected", [
        (torch.add, (4, 3), (4, 3), R, R, R),
        (torch.mul, (4, 3), (4, 3), R, R, R),
        (torch.mm, (4, 3), (3, 5), R, R, R),
        (torch.add, (4, 3), (4, 3), P, P, P),
        (torch.mul, (4, 3), (4, 3), P, R, P),
    ], name_fn=lambda op, x_shape, w_shape, x_type, w_type, expected: (
        f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
    ))
    def test_op(self, op, x_shape, w_shape, x_type, w_type, expected):
        x = self._make_input(x_shape, "tp", x_type)
        w = self._make_input(w_shape, "tp", w_type)
        self.assertEqual(_get_axis_type(op(x, w), "tp"), expected)

    def test_nonlinear_p_rejected(self):
        x = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.relu(x)


instantiate_parametrized_tests(TestGlobalSpmdNoShardAxes)


# =============================================================================
# Ops accepting list of tensors and multi-output ops
# =============================================================================


class TestGlobalSpmdListAndMultiOutput(GlobalSpmdTestCase):
    """Test ops that accept tensor lists (cat, stack) and multi-output ops (sort)."""

    @parametrize("op,shape,typ,expected", [
        (torch.cat, (4, 6), S(1), S(1)),
        (torch.cat, (4, 3), R, R),
        (torch.cat, (4, 3), P, P),
        (torch.cat, (4, 3), I, I),
    ], name_fn=lambda op, shape, typ, expected: f"{op.__name__}_{_type_name(typ)}")
    def test_cat(self, op, shape, typ, expected):
        a = self._make_input(shape, "tp", typ)
        b = self._make_input(shape, "tp", typ)
        self.assertEqual(_get_axis_type(torch.cat([a, b], dim=0), "tp"), expected)

    @parametrize("typ,expected", [
        (S(0), S(1)),
        (R, R),
        (P, P),
    ], name_fn=lambda typ, expected: _type_name(typ))
    def test_stack(self, typ, expected):
        shape = (4, 6) if isinstance(typ, Shard) else (4, 3)
        a = self._make_input(shape, "tp", typ)
        b = self._make_input(shape, "tp", typ)
        self.assertEqual(_get_axis_type(torch.stack([a, b], dim=0), "tp"), expected)

    @parametrize("typ,shape", [
        (R, (4, 3)),
        (S(0), (4, 6)), # global shape (12, 6)
    ], name_fn=lambda typ, shape: _type_name(typ))
    def test_sort(self, typ, shape):
        x = self._make_input(shape, "tp", typ)
        values, indices = torch.sort(x)
        self.assertEqual(_get_axis_type(values, "tp"), typ)
        self.assertEqual(_get_axis_type(indices, "tp"), typ)

    def test_sort_shard_rejected(self):
        x = self._make_input((4, 6), "tp", S(0))
        with self.assertRaises(SpmdTypeError):
            torch.sort(x, dim=0)

    def test_sort_p_rejected(self):
        x = self._make_input((4, 3), "tp", P)
        with self.assertRaises(SpmdTypeError):
            torch.sort(x)


instantiate_parametrized_tests(TestGlobalSpmdListAndMultiOutput)


# =============================================================================
# Multiple independent mesh axes
# =============================================================================


class TestGlobalSpmdMultiAxis(GlobalSpmdTestCase):
    """Test S(i) propagation with multiple independent mesh axes."""

    def test_mm_shard_one_axis_replicate_other(self):
        """S(0)@dp, R@tp propagates independently per axis."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": R})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_mm_shard_both_axes(self):
        """S(0)@dp, S(0)@tp: both axes shard dim 0, propagate independently."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": S(0)})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_mm_different_shard_dims_on_different_axes(self):
        """S(0)@dp on input, S(1)@tp on weight: independent axes, independent shards."""
        x = self._make_multi_axis_input((4, 3), {"dp": S(0), "tp": R})
        w = self._make_multi_axis_input((3, 5), {"dp": R, "tp": S(1)})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, "dp"), S(0))
        self.assertEqual(_get_axis_type(result, "tp"), S(1))


# =============================================================================
# Redistribute with S(i) types
# =============================================================================


class TestGlobalSpmdRedistribute(GlobalSpmdTestCase):
    """Test redistribute with S(i) types and its role in enabling ops."""

    def test_redistribute_s0_to_r(self):
        """redistribute(S(0), R) gathers shards to replicated."""
        x = self._make_input((4, 3), "tp", S(0))
        x_r = redistribute(x, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(x_r, "tp"), R)

    def test_redistribute_r_to_s0(self):
        """redistribute(R, S(0)) shards a replicated tensor."""
        x = self._make_input((6, 3), "tp", R)
        x_s = redistribute(x, "tp", src=R, dst=S(0))
        self.assertEqual(_get_axis_type(x_s, "tp"), S(0))

    def test_redistribute_s0_to_s1(self):
        """redistribute(S(0), S(1)) all-to-all to change shard dim."""
        x = self._make_input((6, 3), "tp", S(0))
        x_s1 = redistribute(x, "tp", src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(x_s1, "tp"), S(1))

    def test_redistribute_enables_mm(self):
        """Redistribute weight from S(0) (K sharded) to R, then mm works.

        Without redistribute, mm(R, S(0)) errors because K is sharded.
        After gathering weight to R, mm(R, R) -> R succeeds.
        """
        x = self._make_input((4, 3), "tp", R)
        # S(0) per-rank (1, 5) → global (3, 5), matching x's K=3
        w = self._make_input((1, 5), "tp", S(0))

        # Direct mm fails: K dim of weight is sharded
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)

        # Redistribute weight: S(0) -> R (all_gather)
        w_r = redistribute(w, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(w_r, "tp"), R)

        # Now mm works: R @ R -> R
        result = torch.mm(x, w_r)
        self.assertEqual(_get_axis_type(result, "tp"), R)

    def test_redistribute_enables_mm_reshard(self):
        """Redistribute weight from S(0) (K sharded) to S(1) (N sharded).

        mm(R, S(1)) is valid: output N dim is sharded -> S(1).
        """
        x = self._make_input((4, 3), "tp", R)
        # S(0) per-rank (1, 6) → global (3, 6). After S(0)→S(1): per-rank (3, 2).
        w = self._make_input((1, 6), "tp", S(0))

        # Redistribute weight: S(0) -> S(1) (all-to-all, K-shard to N-shard)
        w_s1 = redistribute(w, "tp", src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(w_s1, "tp"), S(1))

        # Now mm works: R @ S(1) -> S(1)
        result = torch.mm(x, w_s1)
        self.assertEqual(_get_axis_type(result, "tp"), S(1))

    def test_redistribute_fixes_contracted_dim(self):
        """Both inputs shard K. Direct mm is rejected (implicit Partial).

        x: [M, K@S(1)], w: [K@S(0), N].
        Fix: redistribute x from S(1)->S(0) (shard M), w from S(0)->R.
        Then mm(S(0), R) -> S(0).
        """
        # x: S(1) per-rank (6, 3) → global (6, 9). After S(1)→S(0): (2, 9).
        # w: S(0) per-rank (3, 5) → global (9, 5). After S(0)→R: (9, 5).
        x = self._make_input((6, 3), "tp", S(1))  # K sharded
        w = self._make_input((3, 5), "tp", S(0))  # K sharded

        # Direct mm is rejected (V→P mismatch)
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)

        # Fix: redistribute to avoid contracted sharded dim
        x_s0 = redistribute(x, "tp", src=S(1), dst=S(0))
        w_r = redistribute(w, "tp", src=S(0), dst=R)
        result = torch.mm(x_s0, w_r)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))


# =============================================================================
# Local/global SPMD transition via local_map
# =============================================================================


class TestLocalGlobalTransition(GlobalSpmdTestCase):
    """Test transitions between local and global SPMD via local_map.

    local_map is a higher-order function that:
    - Validates inputs match in_type
    - On entry: decays S(i) -> V for axes with S(i) in in_type
    - Runs the function body in local SPMD (only R, I, V, P propagate)
    - On exit: re-annotates outputs with S(i) from out_type

    in_type is a list of dicts, one per positional arg, specifying the
    expected global SPMD type of each input.
    """

    def test_local_map_matmul(self):
        """Full example from design: local_map wraps a matmul in local SPMD.

        Outside (global SPMD):
          x: S(0)@dp, w: R@dp
        Inside local_map (local SPMD):
          x: V@dp (S decayed), w: R@dp
          x @ w -> V@dp (local SPMD propagation)
        Outside after local_map:
          result: S(0)@dp (from out_type)
        """
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            # Inside local_map, S(0) has decayed to V
            self.assertIs(_get_axis_type(x, "dp"), V)
            self.assertIs(_get_axis_type(w, "dp"), R)
            l1 = x @ w
            # V @ R -> V in local SPMD
            self.assertIs(_get_axis_type(l1, "dp"), V)
            return l1

        y1 = local_matmul(x, w)
        # After local_map, out_type restores S(0)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

    def test_local_map_preserves_non_shard_types(self):
        """R and I types are unchanged across local_map boundary."""
        x = self._make_input((4, 3), "tp", R)

        @local_map(
            in_type=[{"tp": R}],
            out_type={"tp": R},
        )
        def identity(x):
            self.assertIs(_get_axis_type(x, "tp"), R)
            return x.clone()

        y = identity(x)
        self.assertEqual(_get_axis_type(y, "tp"), R)

    def test_local_map_validates_in_type(self):
        """local_map rejects inputs that don't match in_type."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(1)}, {"dp": R}],  # S(1), but x is S(0)
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        with self.assertRaises((SpmdTypeError, AssertionError)):
            local_matmul(x, w)

    def test_local_map_then_global_op(self):
        """After local_map returns to global SPMD, S(i) propagation works."""
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        y1 = local_matmul(x, w)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

        # Back in global SPMD: redistribute then do another op
        y1_r = redistribute(y1, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y1_r, "dp"), R)

        # R @ R -> R in global SPMD
        y2 = y1_r @ w
        self.assertEqual(_get_axis_type(y2, "dp"), R)

    def test_local_map_multiple_outputs(self):
        """local_map with a function returning multiple tensors."""
        x = self._make_input((4, 6), "tp", S(0))

        @local_map(
            in_type=[{"tp": S(0)}],
            out_type={"tp": S(0)},
        )
        def split_fn(x):
            a, b = torch.split(x, 3, dim=1)
            return a, b

        a, b = split_fn(x)
        self.assertEqual(_get_axis_type(a, "tp"), S(0))
        self.assertEqual(_get_axis_type(b, "tp"), S(0))

    def test_round_trip_global_local_global(self):
        """Full round trip: global -> local -> global -> redistribute -> op.

        This mirrors the sample test from the design:
        1. Start with S(0) in global SPMD
        2. Enter local_map: S(0) -> V, do local matmul
        3. Exit local_map: V -> S(0)
        4. In global SPMD: redistribute S(0) -> R
        5. Do global matmul: R @ R -> R
        """
        x = self._make_input((4, 3), "dp", S(0))
        w = self._make_input((3, 5), "dp", R)

        # Step 1-3: local SPMD matmul
        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_matmul(x, w):
            return x @ w

        y1 = local_matmul(x, w)
        self.assertEqual(_get_axis_type(y1, "dp"), S(0))

        # Step 4: redistribute S(0) -> R
        y1_r = redistribute(y1, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y1_r, "dp"), R)

        # Step 5: global matmul R @ R -> R
        y2 = y1_r @ w
        self.assertEqual(_get_axis_type(y2, "dp"), R)

    def test_global_pointwise_chain(self):
        """Chain of pointwise ops in global SPMD preserves S(0).

        R inputs must be redistributed to S(0) before pointwise ops.
        """
        x = self._make_input((4, 3), "tp", S(0))
        # R tensors have global shape (12, 3) to match S(0) global shape
        bias = self._make_input((12, 3), "tp", R)
        scale = self._make_input((12, 3), "tp", R)
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))

        y = x + bias_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))
        y = y * scale_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))
        y = -y
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_matmul_then_pointwise(self):
        """mm(S(0), R) -> S(0), then pointwise ops preserve S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)
        bias = self._make_input((4, 5), "tp", S(0))

        y = torch.mm(x, w)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

        # S(0) + S(0) -> S(0)
        y = y + bias
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_two_matmuls(self):
        """Chain two matmuls: mm(S(0),R) -> S(0), mm(S(0),R) -> S(0)."""
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 5), "tp", R)
        w2 = self._make_input((5, 2), "tp", R)

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        y = torch.mm(h, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_column_then_row_parallel(self):
        """Column-parallel mm(S(0),R) then row-parallel mm(R,S(1)).

        mm(S(0), R) -> S(0), redistribute S(0)->R, mm(R, S(1)) -> S(1).
        """
        x = self._make_input((4, 3), "tp", S(0))
        w1 = self._make_input((3, 6), "tp", R)
        w2 = self._make_input((6, 5), "tp", S(1))

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h_r = redistribute(h, "tp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(h_r, "tp"), R)

        y = torch.mm(h_r, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(1))

    def test_local_map_pointwise_chain(self):
        """local_map wrapping a chain of pointwise ops."""
        x = self._make_input((4, 3), "tp", S(0))
        bias = self._make_input((4, 3), "tp", R)

        @local_map(
            in_type=[{"tp": S(0)}, {"tp": R}],
            out_type={"tp": S(0)},
        )
        def local_fn(x, bias):
            self.assertIs(_get_axis_type(x, "tp"), V)
            self.assertIs(_get_axis_type(bias, "tp"), R)
            y = x + bias  # V + R -> V
            y = -y  # V -> V
            y = y * bias  # V * R -> V
            return y

        result = local_fn(x, bias)
        self.assertEqual(_get_axis_type(result, "tp"), S(0))

    def test_local_global_local(self):
        """local_map -> global ops -> local_map: two local regions around a global region.

        1. local_map: matmul inside local SPMD, output S(0)
        2. Global: redistribute bias R->S(0), pointwise S(0) + S(0)
        3. local_map: second matmul inside local SPMD, output S(0)
        """
        x = self._make_input((4, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        # R bias has global shape (12, 5) to match S(0) output of mm
        bias = self._make_input((12, 5), "dp", R)
        w2 = self._make_input((5, 2), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm1(x, w):
            return x @ w

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm2(x, w):
            return x @ w

        # local region 1
        h = local_mm1(x, w1)
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # global region: redistribute then pointwise
        bias_s = redistribute(bias, "dp", src=R, dst=S(0))
        h = h + bias_s
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # local region 2
        y = local_mm2(h, w2)
        self.assertEqual(_get_axis_type(y, "dp"), S(0))

    def test_local_global_redistribute_local(self):
        """local -> global redistribute -> local: redistribute between two local regions.

        1. local_map: matmul, output S(0)
        2. Global: redistribute S(0) -> R
        3. local_map: second matmul with R input, output R
        """
        x = self._make_input((4, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        w2 = self._make_input((5, 2), "dp", R)

        @local_map(
            in_type=[{"dp": S(0)}, {"dp": R}],
            out_type={"dp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        @local_map(
            in_type=[{"dp": R}, {"dp": R}],
            out_type={"dp": R},
        )
        def local_mm_replicated(x, w):
            return x @ w

        # local region 1: matmul with sharded input
        h = local_mm(x, w1)
        self.assertEqual(_get_axis_type(h, "dp"), S(0))

        # global region: redistribute
        h_r = redistribute(h, "dp", src=S(0), dst=R)
        self.assertEqual(_get_axis_type(h_r, "dp"), R)

        # local region 2: matmul with replicated input
        y = local_mm_replicated(h_r, w2)
        self.assertEqual(_get_axis_type(y, "dp"), R)

    def test_local_map_input_types_restored(self):
        """local_map restores input S types after the call."""
        x = self._make_input((4, 3), "tp", S(0))
        w = self._make_input((3, 5), "tp", R)

        @local_map(
            in_type=[{"tp": S(0)}, {"tp": R}],
            out_type={"tp": S(0)},
        )
        def local_mm(x, w):
            return x @ w

        y = local_mm(x, w)
        # x should still have S(0) after local_map returns
        self.assertEqual(_get_axis_type(x, "tp"), S(0))
        self.assertEqual(_get_axis_type(w, "tp"), R)
        # y should have the out_type
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_mlp_block(self):
        """Simulate an MLP block: mm -> add bias -> relu -> mm -> add bias.

        All ops propagate S(0) through the pipeline.
        """
        x = self._make_input((6, 4), "tp", S(0))
        w1 = self._make_input((4, 8), "tp", R)
        b1 = self._make_input((6, 8), "tp", S(0))
        w2 = self._make_input((8, 3), "tp", R)
        b2 = self._make_input((6, 3), "tp", S(0))

        h = torch.mm(x, w1)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = h + b1
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.relu(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        y = torch.mm(h, w2)
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

        y = y + b2
        self.assertEqual(_get_axis_type(y, "tp"), S(0))

    def test_global_activation_chain(self):
        """Chain of nonlinear activations: abs -> tanh -> sigmoid -> exp."""
        x = self._make_input((6, 4), "tp", S(0))

        h = torch.abs(x)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.tanh(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        h = torch.sigmoid(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

        # exp preserves S(0)
        h = torch.exp(h)
        self.assertEqual(_get_axis_type(h, "tp"), S(0))

    def test_global_transpose_matmul_pattern(self):
        """Transpose weight then matmul: a common pattern in attention.

        w is stored as S(0) on [N, K], transpose to [K, N] -> S(1),
        then mm(R, S(1)) -> S(1).
        """
        x = self._make_input((4, 6), "tp", R)
        w = self._make_input((3, 6), "tp", S(0))

        w_t = torch.t(w)  # S(0) on (3,6) -> S(1) on (6,3)
        self.assertEqual(_get_axis_type(w_t, "tp"), S(1))

        y = torch.mm(x, w_t)
        self.assertEqual(_get_axis_type(y, "tp"), S(1))

    def test_global_normalize_pattern(self):
        """Simulate normalization: x * scale + bias, redistribute R first."""
        x = self._make_input((6, 4), "tp", S(0))
        scale = self._make_input((18, 4), "tp", R)
        bias = self._make_input((18, 4), "tp", R)
        scale_s = redistribute(scale, "tp", src=R, dst=S(0))
        bias_s = redistribute(bias, "tp", src=R, dst=S(0))

        y = x * scale_s + bias_s
        self.assertEqual(_get_axis_type(y, "tp"), S(0))


# =============================================================================
# Backward correctness via global (non-distributed) ground truth
# =============================================================================


class TestGlobalSpmdBackwardGroundTruth(GlobalSpmdTestCase):
    """Verify backward correctness by comparing against non-distributed ground truth.

    Reconstructs global (non-distributed) tensors from per-rank SPMD data,
    runs the same computation on them, and compares gradients. Unlike the
    adjoint identity check, this works for ALL ops including nonlinear.

    Reconstruction rules:
      S(i) -> torch.cat(per-rank, dim=i)
      R/I  -> rank 0's data
      P    -> sum across ranks

    Gradient comparison by input type T (grad has dual type):
      T=S(i): per-rank grad = chunk(global_grad, N, dim=i)
      T=R:    sum(per-rank grads) = global_grad  (grad is P)
      T=P:    each rank grad = global_grad        (grad is R)
      T=I:    each rank grad = global_grad
    """

    def _gradient_type(self, typ):
        if isinstance(typ, Shard):
            return typ
        return {R: P, I: I, V: V, P: R}[typ]

    def _make_random_typed(self, like_lt, typ, axis):
        if isinstance(typ, Shard) or typ is V or typ is P:
            result = self.mode.rank_map(
                lambda r: torch.randn_like(like_lt._local_tensors[r])
            )
        else:
            t = torch.randn_like(like_lt._local_tensors[0])
            result = self.mode.rank_map(lambda r: t.clone())
        assert_type(result, {axis: typ})
        return result

    def _to_global(self, lt, typ):
        """Reconstruct global tensor given LocalTensor and its SPMD type."""
        W = self.WORLD_SIZE
        if isinstance(typ, Shard):
            return torch.cat(
                [lt._local_tensors[r].detach() for r in range(W)], dim=typ.dim
            )
        elif typ is R or typ is I:
            # R/I: all ranks hold the same value
            for r in range(1, W):
                torch.testing.assert_close(
                    lt._local_tensors[0], lt._local_tensors[r],
                    msg=f"{typ} tensor differs at rank {r}",
                )
            return lt._local_tensors[0].detach().clone()
        elif typ is P:
            # P: each rank holds a partial sum, global = sum of partials
            return sum(lt._local_tensors[r].detach() for r in range(W))
        elif typ is V:
            import warnings
            warnings.warn("Cannot reconstruct global tensor from V: skipping gradient comparison")
            return None

        raise ValueError(f"Cannot reconstruct global tensor from {typ}")

    def _gradcheck(self, fn, inputs, input_types, axis):
        """Compare SPMD gradients against global ground truth.

        The same fn must work on both LocalTensors (SPMD) and regular tensors
        (global). This ensures we're testing the exact same computation.

        Args:
            fn: Function that works on both LocalTensors and regular tensors.
            inputs: List of LocalTensor inputs to differentiate.
            input_types: SPMD type for each input.
            axis: Mesh axis name.
        """
        for inp in inputs:
            inp.requires_grad_(True)

        # SPMD forward + backward
        y = fn(*inputs)
        output_type = get_local_type(y).get(axis, V)
        grad_type = self._gradient_type(output_type)
        g = self._make_random_typed(y, grad_type, axis)
        y.backward(g)

        # Reconstruct global tensors and gradient for ground truth computation
        global_inputs = [
            self._to_global(inp, typ)
            for inp, typ in zip(inputs, input_types)
        ]
        global_g = self._to_global(g, grad_type)

        for gi in global_inputs:
            gi.requires_grad_(True)

        # Global forward + backward (SpmdTypeMode passes through for
        # regular tensors without SPMD type annotations)
        global_y = fn(*global_inputs)
        global_y.backward(global_g)

        # Compare gradients: reconstruct SPMD grad using the gradient type
        for inp, gi, typ in zip(inputs, global_inputs, input_types):
            if typ is V:
                continue
            grad_typ = self._gradient_type(typ)
            spmd_grad_global = self._to_global(inp.grad, grad_typ)
            torch.testing.assert_close(spmd_grad_global, gi.grad)

    @parametrize("op,shapes,types", [
        (torch.mm, [(6, 3), (3, 5)], [S(0), R]),
        (torch.mm, [(4, 3), (3, 6)], [R, S(1)]),
        (torch.add, [(6, 3), (6, 3)], [S(0), S(0)]),
        (torch.mul, [(6, 3), (6, 3)], [S(0), S(0)]),
        (torch.neg, [(6, 3)], [S(0)]),
        (torch.relu, [(6, 3)], [S(0)]),
    ], name_fn=lambda op, shapes, types: f"{op.__name__}_{'_'.join(t.name if hasattr(t, 'name') else str(t) for t in types)}")
    def test_op(self, op, shapes, types):
        inputs = [self._make_input(s, "tp", t) for s, t in zip(shapes, types)]
        self._gradcheck(op, inputs, types, "tp")

    def test_mm_relu_mm_chain(self):
        """mm -> relu -> mm: nonlinear in the middle."""
        x = self._make_input((6, 3), "tp", S(0))
        w1 = self._make_input((3, 5), "tp", R)
        w2 = self._make_input((5, 2), "tp", R)

        def fn(x, w1, w2):
            h = torch.mm(x, w1)
            h = torch.relu(h)
            return torch.mm(h, w2)

        self._gradcheck(fn, [x, w1, w2], [S(0), R, R], "tp")

    # -- local/global SPMD mixture --

    def test_local_global_mixture(self):
        """global add -> local_map MLP (mm -> relu -> mm) -> global relu.

        Covers: global pointwise feeding into local_map, nonlinear ops
        inside local_map, and global nonlinear op after local_map.
        """
        axis = "dp"
        input_types = [S(0), R, R]

        x = self._make_input((6, 3), axis, S(0))
        bias = self._make_input((18, 3), axis, R)
        bias_s = redistribute(bias, axis, src=R, dst=S(0))
        w1 = self._make_input((3, 5), axis, R)
        w2 = self._make_input((5, 2), axis, R)
        inputs = [x, w1, w2]

        @local_map(
            in_type=[{axis: S(0)}, {axis: R}, {axis: R}],
            out_type={axis: S(0)},
        )
        def local_mlp(x, w1, w2):
            h = x @ w1
            h = torch.relu(h)
            return h @ w2

        # SPMD forward + backward
        for inp in inputs:
            inp.requires_grad_(True)
        y = torch.relu(local_mlp(x + bias_s, w1, w2))
        output_type = get_local_type(y).get(axis, V)
        grad_type = self._gradient_type(output_type)
        g = self._make_random_typed(y, grad_type, axis)
        # g must be the gradient type of y according to spmd-type definition
        y.backward(g)

        # Ground truth: same math on global tensors
        global_bias = self._to_global(bias_s, S(0))
        global_inputs = [
            self._to_global(inp, typ)
            for inp, typ in zip(inputs, input_types)
        ]
        global_g = self._to_global(g, grad_type)
        for gi in global_inputs:
            gi.requires_grad_(True)
        gx, gw1, gw2 = global_inputs
        h = torch.mm(gx + global_bias, gw1)
        h = torch.relu(h)
        h = torch.relu(torch.mm(h, gw2))
        h.backward(global_g)

        # Compare
        for inp, gi, typ in zip(inputs, global_inputs, input_types):
            grad_typ = self._gradient_type(typ)
            spmd_grad_global = self._to_global(inp.grad, grad_typ)
            torch.testing.assert_close(spmd_grad_global, gi.grad)

    def test_ddp_pattern(self):
        """DDP: data-parallel with replicated weights.

        x: S(0) (batch-sharded); w1, w2: R (replicated)
        Forward: relu(mm(mm(x, w1), w2)) → S(0)
        Backward: x.grad: S(0); w1.grad, w2.grad: P (needs all-reduce in real DDP)
        """
        x = self._make_input((6, 3), "dp", S(0))
        w1 = self._make_input((3, 5), "dp", R)
        w2 = self._make_input((5, 7), "dp", R)

        def fn(x, w1, w2):
            y1 = x @ w1
            y2 = y1 @ w2
            return torch.relu(y2)

        self._gradcheck(fn, [x, w1, w2], [S(0), R, R], "dp")

    def test_megatron_tp_pattern(self):
        """Megatron TP: I→R (no-op fwd, all-reduce bwd), column+row parallel mm.

        1. a: I → redistribute to R (identity fwd, all-reduce bwd)
        2. mm(R, S(1)) → S(1) (column parallel)
        3. local_map: mm(S(1), S(0)) → P (row parallel, contracted dim)
        4. redistribute P → R (all-reduce output)

        a.grad should be I: the backward of redistribute(I→R) all-reduces the
        P gradient (dual of R) into an I gradient (dual of I).
        """
        axis = "tp"
        # DTensor propagation treats local shape as global shape, so use
        # shapes where mm is valid with the given local shapes.
        a = self._make_input((4, 3), axis, I)
        # global shape: (3, 8*tp)
        w_col = self._make_input((3, 8), axis, S(1))
        # global shape: (8*tp, 5)
        w_row = self._make_input((8, 5), axis, S(0))
        inputs = [a, w_col, w_row]
        input_types = [I, S(1), S(0)]
        for inp in inputs:
            inp.requires_grad_(True)

        # Forward: I → R. This should be no-op in the forward, but all-reduce in
        # backward.
        a_r = redistribute(a, axis, src=I, dst=R)
        h = torch.mm(a_r, w_col)  # R @ S(1) → S(1)
        assert_type(h, {axis: S(1)})

        @local_map(
            in_type=[{axis: S(1)}, {axis: S(0)}],
            out_type={axis: P},
        )
        def row_mm(x, w):
            return x @ w

        h_p = row_mm(h, w_row)
        assert_type(h_p, {axis: P})
        y = redistribute(h_p, axis, src=P, dst=I)

        # Backward
        output_type = get_local_type(y).get(axis, V)
        g = self._make_random_typed(y, self._gradient_type(output_type), axis)
        # g must be the gradient type of y according to spmd-type definition
        y.backward(g)

        # Ground truth: same math on plain tensors
        global_inputs = [
            self._to_global(inp, typ)
            for inp, typ in zip(inputs, input_types)
        ]
        global_g = self._to_global(g, self._gradient_type(output_type))
        for gi in global_inputs:
            gi.requires_grad_(True)
        ga, gw_col, gw_row = global_inputs
        global_y = torch.mm(torch.mm(ga, gw_col), gw_row)
        global_y.backward(global_g)

        for inp, gi, typ in zip(inputs, global_inputs, input_types):
            grad_typ = self._gradient_type(typ)
            spmd_grad = self._to_global(inp.grad, grad_typ)
            torch.testing.assert_close(spmd_grad, gi.grad)


instantiate_parametrized_tests(TestGlobalSpmdBackwardGroundTruth)

if __name__ == "__main__":
    run_tests()
