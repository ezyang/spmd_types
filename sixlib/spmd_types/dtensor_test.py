"""Tests for spmd_types._dtensor: DTensor bridge utilities."""

import contextlib
import inspect

import pytest
import torch
import torch.distributed as dist
from sixlib.sharding.common import sharding_to_placements
from sixlib.spmd_types import (
    I,
    P,
    R,
    S,
    spmd_redistribute,
    spmd_type_to_dtensor_placement,
    TensorSharding,
    V,
)
from sixlib.spmd_types._mesh_axis import _reset
from sixlib.test_utils import distributed_test
from sixlib.utils_spmd import current_mesh
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


@pytest.fixture(autouse=True, scope="module")
def _teardown_dist_after_module():
    """Tear down the process group after this module's tests complete.

    Some tests use @distributed_test(backend="fake") which initializes dist
    in the main process without teardown, poisoning later test modules that
    need a different world size.
    """
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class TestSpmdTypeToDtensorPlacement:
    """Tests for spmd_type_to_dtensor_placement."""

    @pytest.mark.parametrize(
        "spmd_type,expected_cls,expected_dim",
        [
            (R, Replicate, None),
            (I, Replicate, None),
            (S(0), Shard, 0),
            (S(1), Shard, 1),
            (S(2), Shard, 2),
            (P, Partial, None),
        ],
        ids=["R", "I", "S0", "S1", "S2", "P"],
    )
    def test_known_types(self, spmd_type, expected_cls, expected_dim):
        result = spmd_type_to_dtensor_placement(spmd_type)
        assert isinstance(result, expected_cls)
        if expected_dim is not None:
            assert result.dim == expected_dim

    def test_varying_raises(self):
        with pytest.raises(ValueError, match="plain Varying"):
            spmd_type_to_dtensor_placement(V)


class TestSpmdDtensorConversions:
    """Tests for spmd_redistribute with DeviceMesh integration."""

    @distributed_test(mesh_dim_sizes={"dp": 1, "tp": 1}, backend="fake")
    def test_multi_axis_redistribute_raises_on_change(self, world_size: int):
        """spmd_redistribute raises when a multi-axis dim changes."""
        mesh = current_mesh()
        dummy = torch.zeros(4, 4)
        dt = DTensor.from_local(dummy, mesh, (Replicate(), Replicate()))

        # ("dp", "tp") -> None: multi-axis dim changes, should raise.
        with pytest.raises(NotImplementedError, match="Multi-axis sharding"):
            spmd_redistribute(
                dt,
                src=TensorSharding(("dp", "tp"), None),
                dst=TensorSharding(None, None),
            )

        # None -> ("dp", "tp"): multi-axis dim changes, should also raise.
        with pytest.raises(NotImplementedError, match="Multi-axis sharding"):
            spmd_redistribute(
                dt,
                src=TensorSharding(None, None),
                dst=TensorSharding(("dp", "tp"), None),
            )

    @distributed_test(mesh_dim_sizes={"dp": 1, "tp": 1}, backend="fake")
    def test_multi_axis_redistribute_ok_when_unchanged(self, world_size: int):
        """spmd_redistribute allows multi-axis dims that don't change (no-op)."""
        mesh = current_mesh()
        dummy = torch.zeros(4, 4)
        dt = DTensor.from_local(dummy, mesh, (Shard(0), Replicate()))

        # Both src and dst have ("dp", "tp") on dim 0 -- no transition, returns x.
        result = spmd_redistribute(
            dt,
            src=TensorSharding(("dp", "tp"), None),
            dst=TensorSharding(("dp", "tp"), None),
        )
        assert result is dt


def _assert_redistribute_matches_dtensor(mesh, global_shape, cases):
    """Run redistribute comparison for a list of (name, src, dst) cases.

    For each case, verifies that spmd_redistribute produces bitwise-identical
    outputs, placements, and gradients compared to DTensor.redistribute.
    """
    for name, src_ts, dst_ts in cases:
        src_pls = sharding_to_placements(src_ts, mesh_dim_names=mesh.mesh_dim_names)
        dst_pls = sharding_to_placements(dst_ts, mesh_dim_names=mesh.mesh_dim_names)

        # Compute local shape from src placements.
        local_shape = list(global_shape)
        for mesh_idx, pl in enumerate(src_pls):
            if isinstance(pl, Shard):
                local_shape[pl.dim] //= mesh.size(mesh_idx)
        local_shape = tuple(local_shape)

        local_data = torch.randn(local_shape)

        # ---- Reference: DTensor.redistribute ----
        ref_local = local_data.clone().detach().requires_grad_(True)
        ref_dt = DTensor.from_local(ref_local, mesh, tuple(src_pls), run_check=False)
        ref_out = ref_dt.redistribute(placements=tuple(dst_pls))
        ref_out_data = ref_out._local_tensor.detach().clone()

        # ---- Test: spmd_redistribute ----
        test_local = local_data.clone().detach().requires_grad_(True)
        test_dt = DTensor.from_local(test_local, mesh, tuple(src_pls), run_check=False)
        test_out = spmd_redistribute(test_dt, src_ts, dst_ts)
        test_out_data = test_out._local_tensor.detach().clone()

        # ---- Backward with explicit ones gradient ----
        ones_grad = DTensor.from_local(
            torch.ones_like(ref_out._local_tensor),
            mesh,
            tuple(dst_pls),
            run_check=False,
        )
        ref_out.backward(ones_grad)
        assert ref_local.grad is not None, f"{name}: ref grad is None"

        test_out.backward(
            DTensor.from_local(
                torch.ones_like(test_out._local_tensor),
                mesh,
                tuple(dst_pls),
                run_check=False,
            )
        )
        assert test_local.grad is not None, f"{name}: test grad is None"

        # ---- Compare ----
        assert torch.equal(ref_out_data, test_out_data), f"{name}: output mismatch"
        assert tuple(ref_out.placements) == tuple(test_out.placements), (
            f"{name}: placement mismatch: {ref_out.placements} vs {test_out.placements}"
        )
        assert torch.equal(ref_local.grad, test_local.grad), (
            f"{name}: grad mismatch "
            f"ref={ref_local.grad.flatten()[:2].tolist()} "
            f"test={test_local.grad.flatten()[:2].tolist()}"
        )


@contextlib.contextmanager
def _local_tensor_env(world_size: int):
    """Set up fake distributed environment with LocalTensorMode."""
    if dist.is_initialized():
        dist.destroy_process_group()
    _reset()
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=0, world_size=world_size, store=store)
    with LocalTensorMode(world_size):
        yield
    if dist.is_initialized():
        dist.destroy_process_group()
    _reset()


class _LocalTensorDTensorMixin:
    """Mixin providing fake PG + LocalTensorMode setup for DTensor tests.

    Subclasses must set WORLD_SIZE (default used when a test does not
    override via the ``local_tensor_env`` fixture).
    """

    WORLD_SIZE: int = 2

    @pytest.fixture(autouse=True)
    def _setup_local_tensor_mode(self):
        """Set up fake distributed environment with LocalTensorMode."""
        with _local_tensor_env(self.WORLD_SIZE):
            yield


class TestSpmdRedistribute(_LocalTensorDTensorMixin):
    """Tests for spmd_redistribute.

    For each (src, dst) TensorSharding transition, verifies that
    spmd_redistribute produces bitwise-identical outputs and gradients
    compared to DTensor.redistribute.
    """

    def test_all_transitions_match_dtensor(self):
        """Verify spmd_redistribute matches DTensor.redistribute.

        Covers single-axis transitions on tp (S/R combinations + all_to_all).
        Compares bitwise-equal outputs, placements, and gradients.
        """
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        global_shape = (4, 8)

        # Each case: (name, src_sharding, dst_sharding).
        # P and I are not expressible via TensorSharding; those transitions
        # use spmd_types.all_reduce / reduce_scatter directly.
        cases = [
            # --- single-axis on tp ---
            ("S0->R", TensorSharding("tp"), TensorSharding()),
            ("S1->R", TensorSharding(None, "tp"), TensorSharding()),
            ("R->S0", TensorSharding(), TensorSharding("tp")),
            ("R->S1", TensorSharding(), TensorSharding(None, "tp")),
            ("S0->S1", TensorSharding("tp"), TensorSharding(None, "tp")),
            ("R->R", TensorSharding(), TensorSharding()),
            ("S0->S0", TensorSharding("tp"), TensorSharding("tp")),
        ]

        _assert_redistribute_matches_dtensor(mesh, global_shape, cases)


class TestSpmdRedistributeMultiAxis(_LocalTensorDTensorMixin):
    """Tests for spmd_redistribute on multi-axis meshes."""

    WORLD_SIZE = 4

    def test_fsdp_tp_transition(self):
        """Verify spmd_redistribute on a fsdp=2, tp=2 mesh."""
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "tp"))
        global_shape = (4, 8)

        cases = [
            # --- multi-axis on fsdp+tp ---
            (
                "fsdp+tp",
                TensorSharding("fsdp", "tp"),
                TensorSharding(),
            ),
        ]

        _assert_redistribute_matches_dtensor(mesh, global_shape, cases)


class TestSpmdRedistributePartialGrad(_LocalTensorDTensorMixin):
    """Tests that spmd_redistribute backward matches DTensor when the
    upstream gradient is Partial (not Replicate).

    Chains spmd_redistribute (S(0)->R all-gather) with a column-parallel
    matmul (weight S(1) on tp).  The matmul backward produces a Partial
    gradient for the all-gather output, exercising the Partial->Replicate
    normalisation path in _FromTorchTensor.backward.
    """

    WORLD_SIZE = 2

    def test_partial_grad_matches_dtensor(self):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))

        src_ts = TensorSharding("tp")  # S(0) on tp
        dst_ts = TensorSharding()  # R

        # Global x: (4, 8), S(0) on tp -> local (2, 8).
        x_local = torch.randn(2, 8)

        # Column-parallel weight: global (8, 4), S(1) on tp -> local (8, 2).
        # mm(x_R, w_S1) output is S(1); backward dx is Partial on tp.
        w_local = torch.randn(8, 2)

        # ---- Reference: DTensor.redistribute ----
        ref_x = x_local.clone().detach().requires_grad_(True)
        ref_dt = DTensor.from_local(ref_x, mesh, (Shard(0),), run_check=False)
        ref_gathered = ref_dt.redistribute(placements=(Replicate(),))
        ref_w = DTensor.from_local(w_local.clone(), mesh, (Shard(1),), run_check=False)
        ref_out = torch.mm(ref_gathered, ref_w)
        ref_out.sum().backward()

        # ---- Test: spmd_redistribute ----
        test_x = x_local.clone().detach().requires_grad_(True)
        test_dt = DTensor.from_local(test_x, mesh, (Shard(0),), run_check=False)
        test_gathered = spmd_redistribute(test_dt, src_ts, dst_ts)
        test_w = DTensor.from_local(w_local.clone(), mesh, (Shard(1),), run_check=False)
        test_out = torch.mm(test_gathered, test_w)
        test_out.sum().backward()

        # ---- Compare ----
        assert ref_x.grad is not None, "ref grad is None"
        assert test_x.grad is not None, "test grad is None"
        assert torch.equal(ref_x.grad, test_x.grad), (
            f"partial grad mismatch: "
            f"ref={ref_x.grad.flatten()[:4].tolist()} "
            f"test={test_x.grad.flatten()[:4].tolist()}"
        )


class TestSpmdAllToAllContiguous(_LocalTensorDTensorMixin):
    """Regression test: spmd_types.all_to_all must handle non-contiguous chunks.

    torch.chunk(x, n, dim=d) produces non-contiguous views when d > 0.
    torch.distributed.all_to_all requires contiguous tensors. This test
    exercises the S(i)->S(j) path to verify that the chunks are made
    contiguous before the collective.
    """

    WORLD_SIZE = 2

    def test_shard_to_shard_forward(self):
        """all_to_all S(0)->S(1) with split on dim 1."""
        from sixlib.spmd_types import all_to_all, assert_type

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        pg = mesh.get_group(0)

        # (4, 4) tensor, S(0)->S(1) splits on dim 1 (non-contiguous chunks).
        x = torch.randn(4, 4)
        assert_type(x, {pg: V})
        result = all_to_all(x, pg, src=S(0), dst=S(1))
        # S(0) input (4, 4) -> S(1) output: split dim 1 by 2, concat dim 0 by 2
        # -> (8, 2)
        assert result.shape == (8, 2), f"got {result.shape}"

    def test_shard_to_shard_backward(self):
        """all_to_all S(0)->S(1) backward."""
        from sixlib.spmd_types import all_to_all, assert_type

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        pg = mesh.get_group(0)

        x = torch.randn(4, 4, requires_grad=True)
        assert_type(x, {pg: V})
        result = all_to_all(x, pg, src=S(0), dst=S(1))
        result.backward(torch.ones_like(result))
        assert x.grad is not None
        assert x.grad.shape == (4, 4)


class TestDtensorPlacementToSpmdType:
    """Tests for dtensor_placement_to_spmd_type."""

    @pytest.mark.parametrize(
        "placement,grad_placement,expected",
        [
            (Replicate(), None, I),
            (Replicate(), Partial(), R),
            (Replicate(), Replicate(), I),
            (Shard(0), None, S(0)),
            (Shard(1), None, S(1)),
            (Shard(2), None, S(2)),
            (Partial(), None, P),
        ],
        ids=[
            "Replicate_default_I",
            "Replicate_grad_Partial_R",
            "Replicate_grad_Replicate_I",
            "S0",
            "S1",
            "S2",
            "Partial",
        ],
    )
    def test_known_placements(self, placement, grad_placement, expected):
        from sixlib.spmd_types import dtensor_placement_to_spmd_type

        result = dtensor_placement_to_spmd_type(placement, grad_placement)
        assert result == expected


class TestDtensorTransparency(_LocalTensorDTensorMixin):
    """Tests for DTensor-transparent SPMD type checking.

    Verifies that:
    - Operations on DTensors pass through without SPMD type errors
    - to_local() on a DTensor produces a local tensor with correct SPMD types
    - from_local() on an annotated local tensor produces a DTensor
    """

    WORLD_SIZE = 2

    def test_dtensor_ops_skip_type_checking(self):
        """Operations on DTensors should pass through without type errors."""
        from sixlib.spmd_types import typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        x_local = torch.randn(4, 4)
        dt = DTensor.from_local(x_local, mesh, (Replicate(),), run_check=False)

        with typecheck():
            # Arithmetic on DTensors should not raise type errors.
            result = dt + dt
            assert isinstance(result, DTensor)

            result2 = dt * 2.0
            assert isinstance(result2, DTensor)

    def test_to_local_sets_spmd_type(self):
        """to_local() should set SPMD type on the resulting local tensor."""
        from sixlib.spmd_types import get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        # Replicate placement, default grad_placements -> I type
        # (DTensor default: grad stays Replicate, no all-reduce in backward)
        x_local = torch.randn(4, 4)
        dt_r = DTensor.from_local(x_local, mesh, (Replicate(),), run_check=False)
        with typecheck():
            local_r = dt_r.to_local()
            lt = get_local_type(local_r)
            assert tp_axis in lt, f"Expected tp axis in local type, got {lt}"
            assert lt[tp_axis] is I, (
                f"Expected I for Replicate (default), got {lt[tp_axis]}"
            )

        # Replicate placement, explicit grad_placements=Partial -> R type
        # (backward does all-reduce)
        with typecheck():
            local_r2 = dt_r.to_local(grad_placements=(Partial(),))
            lt = get_local_type(local_r2)
            assert lt[tp_axis] is R, (
                f"Expected R for Replicate (Partial grad), got {lt[tp_axis]}"
            )

        # Shard(0) placement -> V type (S(0) decays to V as local type)
        dt_s = DTensor.from_local(torch.randn(2, 4), mesh, (Shard(0),), run_check=False)
        with typecheck():
            local_s = dt_s.to_local()
            lt = get_local_type(local_s)
            assert tp_axis in lt, f"Expected tp axis in local type, got {lt}"
            # S(0) is stored as V in the local type system
            assert lt[tp_axis] is V, f"Expected V for Shard(0), got {lt[tp_axis]}"

    def test_from_local_no_spmd_type_on_dtensor(self):
        """from_local() should produce a DTensor without SPMD type annotation."""
        from sixlib.spmd_types import assert_type, get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        with typecheck():
            x = torch.randn(4, 4)
            # Replicate() with default grad_placements is I semantics
            # (backward keeps gradient as Replicate, no all-reduce).
            assert_type(x, {tp_axis: I})
            dt = DTensor.from_local(x, mesh, (Replicate(),), run_check=False)
            # DTensor result should not have SPMD type annotation
            # (DTensor tracks placements via its own metadata).
            lt = get_local_type(dt)
            assert lt == {}, f"Expected empty local type on DTensor, got {lt}"

    @pytest.mark.skipif(
        "grad_placements" not in inspect.signature(DTensor.from_local).parameters,
        reason="DTensor.from_local does not support grad_placements in this PyTorch version",
    )
    def test_from_local_r_with_partial_grad(self):
        """from_local() with Replicate + Partial grad matches R type."""
        from sixlib.spmd_types import assert_type, typecheck
        from sixlib.spmd_types.types import normalize_axis, SpmdTypeError

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        with typecheck():
            x = torch.randn(4, 4)
            assert_type(x, {tp_axis: R})
            # Explicit Partial grad_placements -> R semantics (all-reduce in bwd)
            dt = DTensor.from_local(
                x,
                mesh,
                (Replicate(),),
                run_check=False,
                grad_placements=(Partial(),),
            )
            assert isinstance(dt, DTensor)

    @pytest.mark.skipif(
        "grad_placements" not in inspect.signature(DTensor.from_local).parameters,
        reason="DTensor.from_local does not support grad_placements in this PyTorch version",
    )
    def test_from_local_r_with_default_grad_errors(self):
        """from_local() with Replicate + default grad is I, not R."""
        from sixlib.spmd_types import assert_type, typecheck
        from sixlib.spmd_types.types import normalize_axis, SpmdTypeError

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        with typecheck():
            x = torch.randn(4, 4)
            assert_type(x, {tp_axis: R})
            # Default grad_placements normalizes to Replicate -> I semantics,
            # but tensor is R -> mismatch.
            with pytest.raises(SpmdTypeError):
                DTensor.from_local(x, mesh, (Replicate(),), run_check=False)

    def test_from_local_v_with_shard(self):
        """from_local() with Shard placement matches V type."""
        from sixlib.spmd_types import assert_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        with typecheck():
            x = torch.randn(4, 4)
            assert_type(x, {tp_axis: V})
            dt = DTensor.from_local(x, mesh, (Shard(0),), run_check=False)
            assert isinstance(dt, DTensor)

    def test_from_local_v_with_replicate_errors(self):
        """from_local() with Replicate but V type is a mismatch."""
        from sixlib.spmd_types import assert_type, typecheck
        from sixlib.spmd_types.types import normalize_axis, SpmdTypeError

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        with typecheck():
            x = torch.randn(4, 4)
            assert_type(x, {tp_axis: V})
            with pytest.raises(SpmdTypeError):
                DTensor.from_local(x, mesh, (Replicate(),), run_check=False)

    def test_roundtrip_dtensor_to_local_and_back(self):
        """DTensor -> to_local -> from_local roundtrip preserves types."""
        from sixlib.spmd_types import get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        x_local = torch.randn(4, 4)
        dt = DTensor.from_local(x_local, mesh, (Replicate(),), run_check=False)

        with typecheck():
            # to_local with default grad_placements should annotate with I
            local = dt.to_local()
            lt = get_local_type(local)
            assert lt[tp_axis] is I

            # from_local back to DTensor should work without errors
            dt2 = DTensor.from_local(local, mesh, (Replicate(),), run_check=False)
            assert isinstance(dt2, DTensor)

    def test_redistribute_passes_through(self):
        """DTensor.redistribute() should pass through without type errors.

        _Redistribute is DTensor's internal autograd function for changing
        placements.  It operates on DTensor-internal local tensors and should
        be handled by the generic DTensor passthrough, not as a local autograd
        function.  The actual SPMD types are stamped at the DTensor boundary
        by _ToTorchTensor / _FromTorchTensor.
        """
        from sixlib.spmd_types import get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        x_local = torch.randn(4, 4)
        dt = DTensor.from_local(x_local, mesh, (Shard(0),), run_check=False)

        with typecheck():
            # redistribute Shard(0) -> Replicate should work without errors
            dt_r = dt.redistribute(placements=(Replicate(),))
            assert isinstance(dt_r, DTensor)
            assert dt_r.placements == (Replicate(),)

            # to_local should stamp the correct SPMD type (I for Replicate default)
            local = dt_r.to_local()
            lt = get_local_type(local)
            assert tp_axis in lt, f"Expected tp axis in local type, got {lt}"
            assert lt[tp_axis] is I, (
                f"Expected I for Replicate (default), got {lt[tp_axis]}"
            )

    def test_redistribute_roundtrip_types(self):
        """Redistribute roundtrip: S(0) -> R -> S(0) preserves correct types."""
        from sixlib.spmd_types import get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        tp_axis = normalize_axis(mesh.get_group(0))

        x_local = torch.randn(2, 4)
        dt = DTensor.from_local(x_local, mesh, (Shard(0),), run_check=False)

        with typecheck():
            # S(0) -> R -> S(0)
            dt_r = dt.redistribute(placements=(Replicate(),))
            dt_s = dt_r.redistribute(placements=(Shard(0),))
            assert isinstance(dt_s, DTensor)
            assert dt_s.placements == (Shard(0),)

            # to_local should stamp V (S(0) decays to V)
            local = dt_s.to_local()
            lt = get_local_type(local)
            assert lt[tp_axis] is V, f"Expected V for Shard(0), got {lt[tp_axis]}"

    def test_nested_dtensor_in_list_passes_through(self):
        """torch.cat([dtensor, dtensor]) should pass through without type errors.

        DTensors nested inside a list arg were previously missed by the flat
        isinstance scan in _typecheck_core, causing spurious type errors.
        """
        from sixlib.spmd_types import typecheck

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        dt1 = DTensor.from_local(
            torch.randn(2, 4), mesh, (Replicate(),), run_check=False
        )
        dt2 = DTensor.from_local(
            torch.randn(2, 4), mesh, (Replicate(),), run_check=False
        )

        with typecheck():
            # torch.cat receives [dt1, dt2] as a single list arg.
            result = torch.cat([dt1, dt2], dim=0)
            assert isinstance(result, DTensor)
            assert result.shape == (4, 4)


class TestDtensorTransparencyMultiAxis(_LocalTensorDTensorMixin):
    """Tests for DTensor-transparent SPMD type checking on multi-axis meshes."""

    WORLD_SIZE = 4

    def test_multi_axis_to_local(self):
        """to_local() on a multi-axis mesh sets types for all axes."""
        from sixlib.spmd_types import get_local_type, typecheck
        from sixlib.spmd_types.types import normalize_axis

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        dp_axis = normalize_axis(mesh.get_group(0))
        tp_axis = normalize_axis(mesh.get_group(1))
        dt = DTensor.from_local(
            torch.randn(2, 4),
            mesh,
            (Shard(0), Replicate()),
            run_check=False,
        )

        with typecheck():
            local = dt.to_local()
            lt = get_local_type(local)
            # Shard(0) -> V, Replicate (default grad) -> I
            assert lt[dp_axis] is V, f"Expected V for dp, got {lt[dp_axis]}"
            assert lt[tp_axis] is I, f"Expected I for tp, got {lt[tp_axis]}"
