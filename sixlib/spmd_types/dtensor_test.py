"""Tests for spmd_types._dtensor: DTensor bridge utilities."""

import pytest
import torch
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
from sixlib.test_utils import distributed_test
from sixlib.utils_spmd import current_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
)


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


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
class TestSpmdRedistribute:
    """Tests for spmd_redistribute.

    For each (src, dst) TensorSharding transition, verifies that
    spmd_redistribute produces bitwise-identical outputs and gradients
    compared to DTensor.redistribute.
    """

    @distributed_test(mesh_dim_sizes={"dp": 1, "fsdp": 2, "tp": 2}, timeout=120)
    def test_all_transitions_match_dtensor(self, world_size: int):
        """Verify spmd_redistribute matches DTensor.redistribute.

        Covers single-axis transitions on tp (S/R combinations + all_to_all),
        and a multi-axis transition on fsdp+tp. Compares bitwise-equal outputs,
        placements, and gradients.
        """
        mesh = current_mesh()
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
            # --- multi-axis on fsdp+tp ---
            (
                "fsdp+tp",
                TensorSharding("fsdp", "tp"),
                TensorSharding(),
            ),
        ]

        for name, src_ts, dst_ts in cases:
            src_pls = sharding_to_placements(src_ts, mesh_dim_names=mesh.mesh_dim_names)
            dst_pls = sharding_to_placements(dst_ts, mesh_dim_names=mesh.mesh_dim_names)

            # Compute local shape from src placements.
            local_shape = list(global_shape)
            for mesh_idx, pl in enumerate(src_pls):
                if isinstance(pl, Shard):
                    local_shape[pl.dim] //= mesh.size(mesh_idx)
            local_shape = tuple(local_shape)

            local_data = torch.randn(local_shape, device="cuda")

            # ---- Reference: DTensor.redistribute ----
            ref_local = local_data.clone().detach().requires_grad_(True)
            ref_dt = DTensor.from_local(
                ref_local, mesh, tuple(src_pls), run_check=False
            )
            ref_out = ref_dt.redistribute(placements=tuple(dst_pls))
            ref_out_data = ref_out._local_tensor.detach().clone()

            # ---- Test: spmd_redistribute ----
            test_local = local_data.clone().detach().requires_grad_(True)
            test_dt = DTensor.from_local(
                test_local, mesh, tuple(src_pls), run_check=False
            )
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
                f"{name}: placement mismatch: "
                f"{ref_out.placements} vs {test_out.placements}"
            )
            assert torch.equal(ref_local.grad, test_local.grad), (
                f"{name}: grad mismatch "
                f"ref={ref_local.grad.flatten()[:2].tolist()} "
                f"test={test_local.grad.flatten()[:2].tolist()}"
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
class TestSpmdRedistributePartialGrad:
    """Tests that spmd_redistribute backward matches DTensor when the
    upstream gradient is Partial (not Replicate).

    Chains spmd_redistribute (S(0)->R all-gather) with a column-parallel
    matmul (weight S(1) on tp).  The matmul backward produces a Partial
    gradient for the all-gather output, exercising the Partial->Replicate
    normalisation path in _FromTorchTensor.backward.
    """

    @distributed_test(mesh_dim_sizes={"tp": 2}, timeout=120)
    def test_partial_grad_matches_dtensor(self, world_size: int):
        mesh = current_mesh()

        src_ts = TensorSharding("tp")  # S(0) on tp
        dst_ts = TensorSharding()  # R

        # Global x: (4, 8), S(0) on tp -> local (2, 8).
        x_local = torch.randn(2, 8, device="cuda")

        # Column-parallel weight: global (8, 4), S(1) on tp -> local (8, 2).
        # mm(x_R, w_S1) output is S(1); backward dx is Partial on tp.
        w_local = torch.randn(8, 2, device="cuda")

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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
class TestSpmdAllToAllContiguous:
    """Regression test: spmd_types.all_to_all must handle non-contiguous chunks.

    torch.chunk(x, n, dim=d) produces non-contiguous views when d > 0.
    torch.distributed.all_to_all requires contiguous tensors. This test
    exercises the S(i)->S(j) path on a real NCCL backend to verify
    that the chunks are made contiguous before the collective.
    """

    @distributed_test(mesh_dim_sizes={"tp": 2})
    def test_shard_to_shard_forward(self, world_size: int):
        """all_to_all S(0)->S(1) on real backend with split on dim 1."""
        from sixlib.spmd_types import all_to_all, assert_type

        mesh = current_mesh()
        pg = mesh.get_group(0)

        # (4, 4) tensor, S(0)->S(1) splits on dim 1 (non-contiguous chunks).
        x = torch.randn(4, 4, device="cuda")
        assert_type(x, {pg: V})
        result = all_to_all(x, pg, src=S(0), dst=S(1))
        # S(0) input (4, 4) -> S(1) output: split dim 1 by 2, concat dim 0 by 2
        # -> (8, 2)
        assert result.shape == (8, 2), f"got {result.shape}"

    @distributed_test(mesh_dim_sizes={"tp": 2})
    def test_shard_to_shard_backward(self, world_size: int):
        """all_to_all S(0)->S(1) backward on real backend."""
        from sixlib.spmd_types import all_to_all, assert_type

        mesh = current_mesh()
        pg = mesh.get_group(0)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        assert_type(x, {pg: V})
        result = all_to_all(x, pg, src=S(0), dst=S(1))
        result.backward(torch.ones_like(result))
        assert x.grad is not None
        assert x.grad.shape == (4, 4)
