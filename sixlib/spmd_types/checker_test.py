"""
Tests for SPMD type checker: type inference, strict mode, error messages.

Covers: _checker.py
"""

import unittest

import expecttest
import torch
from sixlib.spmd_types import (
    I,
    P,
    R,
    S,
    Scalar,
    V,
)
from sixlib.spmd_types._checker import (
    assert_type,
    get_axis_local_type,
    has_local_type,
    infer_local_type_for_axis,
    mutate_type,
    OpLinearity,
    SpmdTypeMode,
)
from sixlib.spmd_types._collectives import all_reduce
from sixlib.spmd_types._test_utils import LocalTensorTestCase, SpmdTypeCheckedTestCase
from sixlib.spmd_types.types import SpmdTypeError
from torch.distributed._local_tensor import LocalTensorMode


class TestEinsumTypePropagation(unittest.TestCase):
    """Test einsum type propagation rules.

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestEinsumSingleOperand(unittest.TestCase):
    """Test einsum with single operand (unary operations).

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestLinearTypePropagation(SpmdTypeCheckedTestCase):
    """Test type propagation through F.linear (matmul + optional bias)."""

    def test_linear_r_v_gives_v(self):
        """F.linear with R weight and V input should give V output, not R."""
        inp = self._generate_inputs((2, 4), self.pg, V)
        weight = self._generate_inputs((3, 4), self.pg, R)
        result = torch.nn.functional.linear(inp, weight)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_matmul_r_v_gives_v(self):
        """torch.matmul with R and V should give V output."""
        x = self._generate_inputs((2, 4), self.pg, R)
        y = self._generate_inputs((4, 3), self.pg, V)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_r_v_gives_v(self):
        """torch.add with R and V should give V output."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_p_p_gives_p(self):
        """torch.add with P and P should give P (addition is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_sub_p_p_gives_p(self):
        """torch.sub with P and P should give P (subtraction is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.sub(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_matmul_p_rejected(self):
        """torch.matmul with all-P is rejected (matmul is not linear in this sense)."""
        x = self._generate_inputs((2, 4), self.pg, P)
        y = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.matmul(x, y)

    def test_mul_p_r_gives_p(self):
        """torch.mul with P and R should give P (multilinear: P in one factor, R in other)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.mul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_cat_p_p_gives_p(self):
        """torch.cat with P and P should give P (cat is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.cat([x, y])
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_matmul_p_r_gives_p(self):
        """torch.matmul with P and R should give P (multilinear: P in one factor)."""
        x = self._generate_inputs((2, 4), self.pg, P)
        y = self._generate_inputs((4, 3), self.pg, R)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_p_r_rejected(self):
        """torch.add with mixed P and R is still rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_add_p_scalar_rejected(self):
        """torch.add(P, scalar) is affine, not linear -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1.0)

    def test_sub_p_scalar_rejected(self):
        """torch.sub(P, scalar) is affine -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.sub(x, 1.0)

    def test_add_p_int_scalar_rejected(self):
        """torch.add(P, int_scalar) is also affine -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1)

    def test_cat_p_with_dim_arg_allowed(self):
        """torch.cat([P, P], 0) -- dim is not a tensor input, should be allowed."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.cat([x, y], 0)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_mul_p_scalar_allowed(self):
        """torch.mul(P, scalar) -- multilinear, scalar ignored, P propagates."""
        x = self._generate_inputs((4,), self.pg, P)
        result = torch.mul(x, 2.0)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_div_i_scalar_gives_i(self):
        """I / scalar should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = x / 2.0
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_add_i_scalar_gives_i(self):
        """torch.add(I, scalar) should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = torch.add(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_sub_i_scalar_gives_i(self):
        """torch.sub(I, scalar) should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = torch.sub(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_add_v_scalar_gives_v(self):
        """torch.add(V, scalar) should give V (scalar adopts V type)."""
        x = self._generate_inputs((4,), self.pg, V)
        result = torch.add(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_mul_v_scalar_gives_v(self):
        """torch.mul(V, scalar) should give V (scalar adopts V type)."""
        x = self._generate_inputs((4,), self.pg, V)
        result = torch.mul(x, 2.0)
        self.assertIs(get_axis_local_type(result, self.pg), V)


class TestStrictMode(SpmdTypeCheckedTestCase):
    """Test SpmdTypeMode strict mode which errors on unannotated tensors."""

    def setUp(self):
        """Enter LocalTensorMode and strict SpmdTypeMode for each test."""
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self.spmd_mode = SpmdTypeMode(strict_mode="strict")
        self.spmd_mode.__enter__()

    def test_strict_mixed_annotated_unannotated_fails(self):
        """Strict mode raises when one operand is annotated and the other is not."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_mixed_cat_fails(self):
        """Strict mode catches mixed typed/untyped tensors inside lists (e.g. torch.cat)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.cat([x, y])

    def test_strict_all_annotated_passes(self):
        """Strict mode allows operations when all tensors are annotated."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_strict_all_v_annotated_mixed_with_unannotated_fails(self):
        """Strict mode catches all-V tensor mixed with unannotated."""
        x = self._generate_inputs((4,), self.pg, V)
        # x has _spmd_types attr (set to {self.pg: V})
        y = self.rank_map(lambda r: torch.randn(4))
        # y has no _spmd_types attr at all
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_all_unannotated_fails(self):
        """Strict mode raises when any tensor is unannotated, even if all are."""
        x = self.rank_map(lambda r: torch.randn(4))
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_collective_typed_input_passes(self):
        """Strict mode allows collectives when the input tensor is typed."""
        x = self._generate_inputs((4,), self.pg, P)
        result = all_reduce(x, self.pg, src=P, dst=R)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_strict_collective_untyped_input_fails(self):
        """Strict mode raises when a collective receives an unannotated tensor."""
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            all_reduce(y, self.pg, src=P, dst=R)

    def test_strict_out_kwarg_unannotated_passes(self):
        """Strict mode allows unannotated out= tensor (it's just a pre-allocated destination)."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))  # unannotated
        self.assertFalse(has_local_type(c))
        torch.add(a, b, out=c)  # should NOT raise
        # out= tensor gets the inferred type
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_strict_out_kwarg_inputs_still_checked(self):
        """Strict mode still checks actual input operands even when out= is present."""
        a = self.rank_map(lambda r: torch.randn(4))  # unannotated input
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(a, b, out=c)

    def test_nonstrict_mixed_passes(self):
        """Non-strict mode skips type inference when inputs are mixed."""
        self.spmd_mode.__exit__(None, None, None)
        nonstrict = SpmdTypeMode(strict_mode="permissive")
        nonstrict.__enter__()
        try:
            x = self._generate_inputs((4,), self.pg, R)
            y = self.rank_map(lambda r: torch.randn(4))
            result = torch.add(x, y)  # Should not raise
            # Result is untyped: inferring from partial inputs could produce
            # incorrect types (e.g., R when a hidden V would make it V).
            self.assertFalse(has_local_type(result))
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()


class TestFactoryTensors(LocalTensorTestCase):
    """Test factory tensor propagation in strict mode.

    Factory tensors are created by ops with no typed tensor inputs (e.g.,
    torch.zeros, torch.ones).  They carry a factory marker that propagates
    through operations on other factory tensors, deferring the real type
    annotation until the tensor meets a real-typed tensor.
    """

    def setUp(self):
        super().setUp()
        self.spmd_mode = SpmdTypeMode(strict_mode="strict")
        self.spmd_mode.__enter__()

    def tearDown(self):
        self.spmd_mode.__exit__(None, None, None)
        super().tearDown()

    def test_factory_creates_factory_tensor(self):
        """A factory op in strict mode marks the result as factory."""
        from sixlib.spmd_types._type_attr import is_factory

        z = torch.zeros(4)
        self.assertTrue(is_factory(z))
        self.assertFalse(has_local_type(z))

    def test_factory_propagates_through_ops(self):
        """Ops on factory tensors produce factory results."""
        from sixlib.spmd_types._type_attr import is_factory

        a = torch.zeros(4)
        b = torch.ones(4)
        c = a + b
        self.assertTrue(is_factory(c))
        self.assertFalse(has_local_type(c))

    def test_factory_meets_typed_raises(self):
        """Mixing a factory tensor with a typed tensor raises."""
        x = self._generate_inputs((4,), self.pg, R)
        z = torch.zeros(4)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, z)

    def test_factory_annotated_then_used(self):
        """assert_type on a factory tensor clears the marker and sets the type."""
        from sixlib.spmd_types._type_attr import is_factory

        z = torch.zeros(4)
        self.assertTrue(is_factory(z))
        assert_type(z, {self.pg: R})
        self.assertFalse(is_factory(z))
        self.assertTrue(has_local_type(z))
        self.assertIs(get_axis_local_type(z, self.pg), R)

    def test_factory_annotated_combines_with_typed(self):
        """Once annotated, a former factory tensor combines with typed tensors."""
        x = self._generate_inputs((4,), self.pg, R)
        z = torch.zeros(4)
        assert_type(z, {self.pg: R})
        result = torch.add(x, z)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_truly_untyped_still_errors(self):
        """Truly untyped tensors (created outside the mode) still error."""
        a = self.rank_map(lambda r: torch.randn(4))
        b = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(a, b)

    def test_factory_error_message_says_factory(self):
        """Error message distinguishes factory from unannotated tensors."""
        x = self._generate_inputs((4,), self.pg, R)
        z = torch.zeros(4)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.add(x, z)
        self.assertIn("factory", str(ctx.exception))

    def test_strict_factory_mode_errors_at_creation(self):
        """strict_factory=True errors immediately on factory creation."""
        self.spmd_mode.__exit__(None, None, None)
        strict_factory_mode = SpmdTypeMode(strict_mode="strict_factory")
        strict_factory_mode.__enter__()
        try:
            with self.assertRaises(SpmdTypeError) as ctx:
                torch.zeros(4)
            self.assertIn("Strict factory mode", str(ctx.exception))
        finally:
            strict_factory_mode.__exit__(None, None, None)
            self.spmd_mode.__enter__()

    def test_factory_chain_then_annotate(self):
        """A chain of factory ops produces factory, then assert_type works."""
        from sixlib.spmd_types._type_attr import is_factory

        a = torch.zeros(4)
        b = torch.ones(4)
        c = a + b
        d = c * 2.0
        self.assertTrue(is_factory(d))
        assert_type(d, {self.pg: V})
        self.assertFalse(is_factory(d))
        self.assertIs(get_axis_local_type(d, self.pg), V)

    def test_nonstrict_factory_propagates(self):
        """Factory propagates in non-strict mode too."""
        from sixlib.spmd_types._type_attr import is_factory

        self.spmd_mode.__exit__(None, None, None)
        nonstrict = SpmdTypeMode(strict_mode="permissive")
        nonstrict.__enter__()
        try:
            z = torch.zeros(4)
            self.assertTrue(is_factory(z))
            w = z + torch.ones(4)
            self.assertTrue(is_factory(w))
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()

    def test_nonstrict_factory_plus_typed_no_error(self):
        """Non-strict skips inference when factory is mixed with typed."""
        self.spmd_mode.__exit__(None, None, None)
        nonstrict = SpmdTypeMode(strict_mode="permissive")
        nonstrict.__enter__()
        try:
            x = self._generate_inputs((4,), self.pg, R)
            z = torch.zeros(4)
            result = torch.add(x, z)  # Should not raise
            # Result is untyped: partial inputs could produce wrong types.
            self.assertFalse(has_local_type(result))
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()

    def test_nonstrict_truly_untyped_not_promoted_to_factory(self):
        """In non-strict mode, truly untyped tensors must not become factory."""
        from sixlib.spmd_types._type_attr import is_factory

        self.spmd_mode.__exit__(None, None, None)
        # Create tensor outside any SpmdTypeMode -- truly untyped.
        a = torch.randn(4)
        nonstrict = SpmdTypeMode(strict_mode="permissive")
        nonstrict.__enter__()
        try:
            self.assertFalse(is_factory(a))
            self.assertFalse(has_local_type(a))
            # Op on truly untyped input: result stays truly untyped.
            b = a + 1.0
            self.assertFalse(is_factory(b))
            self.assertFalse(has_local_type(b))
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()

    def test_nonstrict_factory_plus_truly_untyped_stays_untyped(self):
        """In non-strict mode, factory + truly untyped must not produce factory."""
        from sixlib.spmd_types._type_attr import is_factory

        self.spmd_mode.__exit__(None, None, None)
        # Create tensor outside any SpmdTypeMode -- truly untyped.
        a = torch.randn(4)
        nonstrict = SpmdTypeMode(strict_mode="permissive")
        nonstrict.__enter__()
        try:
            z = torch.zeros(4)  # factory (created inside mode)
            self.assertTrue(is_factory(z))
            self.assertFalse(is_factory(a))
            result = a + z
            # Truly untyped contaminates: result must not be factory.
            self.assertFalse(is_factory(result))
            self.assertFalse(has_local_type(result))
        finally:
            nonstrict.__exit__(None, None, None)
            self.spmd_mode.__enter__()

    def test_factory_spmd_collective_raises(self):
        """Passing a factory tensor to an SPMD collective raises."""
        z = torch.zeros(4)
        with self.assertRaises(SpmdTypeError) as ctx:
            all_reduce(z, self.pg, dst=R)
        self.assertIn("factory", str(ctx.exception))

    def test_factory_raw_collective_raises(self):
        """Passing a factory tensor to a raw dist collective raises."""
        import torch.distributed as dist

        pg = self.pg
        z = torch.zeros(4)
        output = torch.empty(self.WORLD_SIZE * 4)
        with self.assertRaises(SpmdTypeError) as ctx:
            dist.all_gather_into_tensor(output, z, group=pg)
        self.assertIn("factory", str(ctx.exception))


class TestTypeErrorMessages(expecttest.TestCase):
    """Test that type errors include actionable fix suggestions.

    Tests call infer_local_type_for_axis directly rather than going through
    torch ops to test specific axis-type combinations in isolation.
    """

    def test_general_I_R(self):
        """I + R: suggest reinterpret I->R (I->V filtered: changes output from R to V)."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_V(self):
        """I + V: suggest reinterpret I->R or I->V."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  reinterpret(tensor, 'tp', src=I, dst=V) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_P(self):
        """I + P: suggest convert I->P or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, P])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )

    def test_general_P_R(self):
        """P + R: suggest all_reduce P->R or convert R->P."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' cannot propagate through non-linear ops (non-linear of a partial sum != partial sum of non-linear). Reduce first with all_reduce or reduce_scatter. Found types: [P, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_P_V(self):
        """P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_R_P_V(self):
        """R + P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [R, P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [R, P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_I_R_V(self):
        """I + R + V: suggest reinterpret I->R or I->V."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [I, R, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [I, R, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  reinterpret(tensor, 'tp', src=I, dst=V) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_mul_P_P(self):
        """P * P: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial in multiple factors of multilinear op on axis 'tp' is forbidden. Reduce all but one factor first. Found types: [P, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_V(self):
        """P * V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, V], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis 'tp' cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, 'tp', src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_I(self):
        """P * I: suggest reinterpret I->R or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis("tp", [P, I], linearity=OpLinearity.MULTILINEAR)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis 'tp' cannot mix with other types. Found types: [P, I]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, 'tp', src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  all_reduce(tensor, 'tp', src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )


class TestOpLinearity(unittest.TestCase):
    """Test the OpLinearity parameter on infer_local_type_for_axis."""

    def test_all_p_linear_allowed(self):
        """All-P with LINEAR should return P."""
        result = infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.LINEAR)
        self.assertIs(result, P)

    def test_all_p_nonlinear_rejected(self):
        """All-P with NONLINEAR (default) should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, P])

    def test_mixed_p_r_rejected_even_linear(self):
        """Mixed P+R is rejected even with LINEAR."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, R], linearity=OpLinearity.LINEAR)

    def test_all_r_unaffected_by_linearity(self):
        """All-R works regardless of linearity."""
        self.assertIs(infer_local_type_for_axis("tp", [R, R]), R)
        self.assertIs(
            infer_local_type_for_axis("tp", [R, R], linearity=OpLinearity.LINEAR), R
        )

    def test_multilinear_p_r_gives_p(self):
        """MULTILINEAR with P and R should return P."""
        result = infer_local_type_for_axis(
            "tp", [P, R], linearity=OpLinearity.MULTILINEAR
        )
        self.assertIs(result, P)

    def test_multilinear_p_p_rejected(self):
        """MULTILINEAR with P*P should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, P], linearity=OpLinearity.MULTILINEAR)

    def test_multilinear_p_v_rejected(self):
        """MULTILINEAR with P and V should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis("tp", [P, V], linearity=OpLinearity.MULTILINEAR)


class TestAssertTypeShardSugar(unittest.TestCase):
    """Test assert_type with S(i) sugar and PartitionSpec."""

    def test_shard_sugar_sets_varying(self):
        """S(i) in types should set the axis to V in local types."""
        x = torch.randn(4)
        assert_type(x, {"tp": S(0)})
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_shard_sugar_duplicate_dim_rejected(self):
        """S(i) entries mapping multiple axes to the same dim should be rejected."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"dp": S(0), "tp": S(0)})

    def test_shard_sugar_negative_dim(self):
        """S(-1) should resolve to the last dimension."""
        x = torch.randn(3, 4)
        assert_type(x, {"tp": S(-1)})
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_shard_sugar_with_partition_spec_rejected(self):
        """Cannot mix S(i) in types with an explicit partition_spec."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(TypeError):
            assert_type(x, {"tp": S(1)}, PartitionSpec(None, "tp"))

    def test_partition_spec_sets_varying(self):
        """Axes in partition_spec should be treated as Varying."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {}, PartitionSpec(None, "tp"))
        self.assertIs(get_axis_local_type(x, "tp"), V)

    def test_partition_spec_wrong_length_rejected(self):
        """PartitionSpec length must match tensor ndim."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {}, PartitionSpec("tp"))

    def test_partition_spec_conflicts_with_non_varying_rejected(self):
        """Axis in partition_spec that is R/I/P in types should be rejected."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": R}, PartitionSpec(None, "tp"))

    def test_partition_spec_with_explicit_v_ok(self):
        """Explicit V in types for an axis in partition_spec should be allowed."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {"dp": V}, PartitionSpec(None, None))
        self.assertIs(get_axis_local_type(x, "dp"), V)

    def test_mixed_shard_and_local_types(self):
        """S(i) for one axis and R/I/P for another should work."""
        x = torch.randn(3, 4)
        assert_type(x, {"dp": S(0), "tp": I})
        self.assertIs(get_axis_local_type(x, "dp"), V)
        self.assertIs(get_axis_local_type(x, "tp"), I)

    def test_partition_spec_with_non_varying_types(self):
        """partition_spec with R/I axes in types should work if they don't overlap."""
        from sixlib.spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {"tp": I}, PartitionSpec("dp", None))
        self.assertIs(get_axis_local_type(x, "tp"), I)
        self.assertIs(get_axis_local_type(x, "dp"), V)

    def test_shard_sugar_out_of_bounds_positive(self):
        """S(i) with i >= ndim should be rejected."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(2)})

    def test_shard_sugar_out_of_bounds_negative(self):
        """S(i) with i < -ndim should be rejected."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(-3)})

    def test_shard_sugar_0d_tensor_rejected(self):
        """S(i) on a 0-d tensor should be rejected."""
        x = torch.tensor(1.0)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {"tp": S(0)})


class TestOpLinearityRegistry(SpmdTypeCheckedTestCase):
    """Test that the expanded _OP_REGISTRY covers aliases, dunders, and structural ops."""

    def test_clone_functional_form(self):
        """torch.clone(P) should give P (functional form, not just Tensor.clone)."""
        x = self._generate_inputs((4,), self.pg, P)
        result = torch.clone(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_concat_aliases(self):
        """torch.concat and torch.concatenate should propagate P like torch.cat."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        for fn in [torch.concat, torch.concatenate]:
            result = fn([x, y])
            self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_structural_ops_preserve_p(self):
        """Structural ops (reshape, view, transpose, etc.) should preserve P."""
        x = self._generate_inputs((2, 3), self.pg, P)
        # NB: contiguous is registered but untestable here -- LocalTensorMode
        # decomposes it to per-rank calls that bypass SpmdTypeMode.
        ops = [
            ("reshape", lambda t: torch.reshape(t, (6,))),
            ("Tensor.view", lambda t: t.view(6)),
            ("transpose", lambda t: torch.transpose(t, 0, 1)),
            ("squeeze", lambda t: torch.squeeze(t.unsqueeze(0))),
            ("unsqueeze", lambda t: torch.unsqueeze(t, 0)),
            ("flatten", lambda t: torch.flatten(t)),
            ("expand", lambda t: t.unsqueeze(0).expand(2, 2, 3)),
            ("permute", lambda t: torch.permute(t, (1, 0))),
        ]
        for name, op in ops:
            result = op(x)
            self.assertIs(
                get_axis_local_type(result, self.pg),
                P,
                f"{name} did not preserve P type",
            )

    def test_neg_preserves_p(self):
        """torch.neg(P) and -P (unary neg operator) should both give P."""
        x = self._generate_inputs((4,), self.pg, P)
        result_func = torch.neg(x)
        self.assertIs(get_axis_local_type(result_func, self.pg), P)
        result_op = -x
        self.assertIs(get_axis_local_type(result_op, self.pg), P)

    def test_operator_add_p_p(self):
        """P + P via the + operator should give P."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = x + y
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_split_chunk_unbind_preserve_p(self):
        """Multi-output ops (split, chunk, unbind) should propagate P."""
        x = self._generate_inputs((6,), self.pg, P)
        for name, fn in [
            ("split", lambda t: torch.split(t, 3)),
            ("chunk", lambda t: torch.chunk(t, 2)),
            ("unbind", lambda t: torch.unbind(t.view(2, 3))),
        ]:
            results = fn(x)
            for i, r in enumerate(results):
                self.assertIs(
                    get_axis_local_type(r, self.pg),
                    P,
                    f"{name}[{i}] did not preserve P type",
                )

    def test_inplace_add_preserves_p(self):
        """P.add_(P) should give P."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), P)

    def test_div_p_r_gives_p(self):
        """torch.div(P, R) should give P (linear in numerator when denominator is fixed)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_div_r_p_rejected(self):
        """torch.div(R, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_p_p_rejected(self):
        """torch.div(P, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_r_v_gives_v(self):
        """torch.div(R, V) should give V (normal non-P path, no fixed_args filtering)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_registered_ops_are_callable(self):
        """Every key in _OP_REGISTRY should be a callable (no stale references)."""
        from sixlib.spmd_types._checker import _OP_REGISTRY

        for func in _OP_REGISTRY:
            self.assertTrue(
                callable(func),
                f"Registry entry {func!r} is not callable",
            )

    def test_all_decomp_rules_are_callable(self):
        """Every key in _DECOMP_TYPE_RULES should be a callable."""
        from sixlib.spmd_types._checker import _DECOMP_TYPE_RULES

        for func in _DECOMP_TYPE_RULES:
            self.assertTrue(
                callable(func),
                f"Decomp rule entry {func!r} is not callable",
            )


class TestAddmmTypeDecomposition(SpmdTypeCheckedTestCase):
    """Test type-level decomposition for addmm and related ops.

    addmm(self, mat1, mat2) = self + mm(mat1, mat2) decomposes into a sum
    (LINEAR) of self and a matmul (MULTILINEAR) of mat1 and mat2. This catches
    bugs where e.g. addmm(P, R, R) was incorrectly accepted as P.
    """

    # --- addmm ---

    def test_addmm_P_R_R_rejected(self):
        """addmm(P, R, R) must error: R@R=R, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_P_R_rejected(self):
        """addmm(R, P, R) must error: P@R=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_R_P_rejected(self):
        """addmm(R, R, P) must error: R@P=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_P_P_R_gives_P(self):
        """addmm(P, P, R): mm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmm_P_R_P_gives_P(self):
        """addmm(P, R, P): mm(R,P)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmm_R_R_R_gives_R(self):
        """addmm(R, R, R) -> R (no change from before)."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_addmm_V_R_R_gives_V(self):
        """addmm(V, R, R): mm(R,R)=R, then V+R=V."""
        self_t = self._generate_inputs((2, 3), self.pg, V)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_addmm_R_P_P_rejected(self):
        """addmm(R, P, P): mm(P,P) is invalid (two P factors in multilinear)."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_V_V_V_gives_V(self):
        """addmm(V, V, V): mm(V,V)=V, then V+V=V."""
        self_t = self._generate_inputs((2, 3), self.pg, V)
        mat1 = self._generate_inputs((2, 4), self.pg, V)
        mat2 = self._generate_inputs((4, 3), self.pg, V)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    # --- addmv ---

    def test_addmv_P_R_R_rejected(self):
        """addmv(P, R, R) must error."""
        self_t = self._generate_inputs((3,), self.pg, P)
        mat = self._generate_inputs((3, 4), self.pg, R)
        vec = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmv(self_t, mat, vec)

    def test_addmv_P_P_R_gives_P(self):
        """addmv(P, P, R): mv(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3,), self.pg, P)
        mat = self._generate_inputs((3, 4), self.pg, P)
        vec = self._generate_inputs((4,), self.pg, R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmv_R_R_R_gives_R(self):
        """addmv(R, R, R) -> R."""
        self_t = self._generate_inputs((3,), self.pg, R)
        mat = self._generate_inputs((3, 4), self.pg, R)
        vec = self._generate_inputs((4,), self.pg, R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- baddbmm ---

    def test_baddbmm_P_R_R_rejected(self):
        """baddbmm(P, R, R) must error."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, P)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, R)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.baddbmm(self_t, batch1, batch2)

    def test_baddbmm_P_P_R_gives_P(self):
        """baddbmm(P, P, R): bmm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, P)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, P)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_baddbmm_R_R_R_gives_R(self):
        """baddbmm(R, R, R) -> R."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, R)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, R)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- addr ---

    def test_addr_P_R_R_rejected(self):
        """addr(P, R, R) must error."""
        self_t = self._generate_inputs((3, 4), self.pg, P)
        vec1 = self._generate_inputs((3,), self.pg, R)
        vec2 = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addr(self_t, vec1, vec2)

    def test_addr_P_P_R_gives_P(self):
        """addr(P, P, R): outer(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3, 4), self.pg, P)
        vec1 = self._generate_inputs((3,), self.pg, P)
        vec2 = self._generate_inputs((4,), self.pg, R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addr_R_R_R_gives_R(self):
        """addr(R, R, R) -> R."""
        self_t = self._generate_inputs((3, 4), self.pg, R)
        vec1 = self._generate_inputs((3,), self.pg, R)
        vec2 = self._generate_inputs((4,), self.pg, R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- sum/mean (LINEAR reductions) ---

    def test_sum_P_gives_P(self):
        """torch.sum(P) -> P (sum is linear)."""
        x = self._generate_inputs((2, 3), self.pg, P)
        result = torch.sum(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_mean_P_gives_P(self):
        """torch.mean(P) -> P (mean is linear)."""
        # randn already produces float32 so torch.mean works directly
        x = self._generate_inputs((2, 3), self.pg, P)
        result = torch.mean(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)


class TestScalarWrapper(SpmdTypeCheckedTestCase):
    """Test the Scalar wrapper for annotating Python scalars with SPMD types."""

    def test_scalar_has_local_type(self):
        """has_local_type(Scalar(1.0, {tp: R})) is True."""
        s = Scalar(1.0, {"tp": R})
        self.assertTrue(has_local_type(s))

    def test_scalar_get_local_type(self):
        """get_local_type returns the stored type."""
        s = Scalar(1.0, {"tp": R})
        self.assertEqual(get_axis_local_type(s, "tp"), R)

    def test_add_r_scalar_v(self):
        """torch.add(R, Scalar(V)) -> V.

        With a plain 1.0, R + scalar -> R. With Scalar(1.0, {tp: V}),
        R + V -> V. This catches rank-dependent scalars.
        """
        x = self._generate_inputs((4,), self.pg, R)
        s = Scalar(1.0, {self.pg: V})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_v_scalar_v(self):
        """torch.add(V, Scalar(V)) -> V."""
        x = self._generate_inputs((4,), self.pg, V)
        s = Scalar(1.0, {self.pg: V})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_mul_p_scalar_r(self):
        """torch.mul(P, Scalar(R)) -> P (multilinear)."""
        x = self._generate_inputs((4,), self.pg, P)
        s = Scalar(2.0, {self.pg: R})
        result = torch.mul(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_p_scalar_p(self):
        """torch.add(P, Scalar(P)) -> P (linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        s = Scalar(1.0, {self.pg: P})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_r_scalar_p_error(self):
        """torch.add(R, Scalar(P)) -> SpmdTypeError (P+R is affine).

        With a plain 1.0, R + scalar -> R. With Scalar(local_sum, {tp: P}),
        R + P is invalid (affine, not linear).
        """
        x = self._generate_inputs((4,), self.pg, R)
        s = Scalar(1.0, {self.pg: P})
        with self.assertRaises(SpmdTypeError):
            torch.add(x, s)

    def test_add_i_scalar_v_error(self):
        """torch.add(I, Scalar(V)) -> SpmdTypeError (I can't mix)."""
        x = self._generate_inputs((4,), self.pg, I)
        s = Scalar(1.0, {self.pg: V})
        with self.assertRaises(SpmdTypeError):
            torch.add(x, s)

    def test_scalar_without_type_mode(self):
        """Scalar unwraps and computes without SpmdTypeMode."""
        # Exit the SpmdTypeMode to test Scalar outside type checking.
        self.spmd_mode.__exit__(None, None, None)
        try:
            x = torch.tensor([1.0, 2.0, 3.0])
            s = Scalar(2.0, {"tp": V})
            result = torch.mul(x, s)
            expected = torch.tensor([2.0, 4.0, 6.0])
            torch.testing.assert_close(result, expected)
        finally:
            self.spmd_mode.__enter__()


class TestMutationTypeChecking(SpmdTypeCheckedTestCase):
    """Test that in-place/out operations validate SPMD type consistency.

    Mutating operations (trailing underscore, out= kwarg, result-is-input
    identity) must not silently change a tensor's SPMD type. For example,
    x.add_(y) where x=R and y=V would infer V output, but since x is R
    and is being mutated, this is an error.
    """

    # --- In-place ops that preserve type (should pass) ---

    def test_inplace_add_r_r_ok(self):
        """R.add_(R) -> R: type preserved, no error."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, R)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), R)

    def test_inplace_add_v_v_ok(self):
        """V.add_(V) -> V: type preserved, no error."""
        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, V)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_add_v_r_ok(self):
        """V.add_(R) -> V: output is V, self is V, type preserved."""
        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, R)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_mul_v_scalar_ok(self):
        """V.mul_(2.0) -> V: type preserved."""
        x = self._generate_inputs((4,), self.pg, V)
        x.mul_(2.0)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_neg_r_ok(self):
        """R.neg_() -> R: type preserved."""
        x = self._generate_inputs((4,), self.pg, R)
        x.neg_()
        self.assertIs(get_axis_local_type(x, self.pg), R)

    # --- In-place ops that would change type (should error) ---

    def test_inplace_add_r_v_rejected(self):
        """R.add_(V): output would be V but self is R. Error."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        with self.assertRaises(SpmdTypeError) as ctx:
            x.add_(y)
        self.assertIn("in-place/out", str(ctx.exception))

    def test_inplace_mul_r_v_rejected(self):
        """R.mul_(V): output would be V but self is R. Error."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        with self.assertRaises(SpmdTypeError) as ctx:
            x.mul_(y)
        self.assertIn("in-place/out", str(ctx.exception))

    def test_iadd_r_v_rejected(self):
        """R += V via __iadd__: should also be caught."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        with self.assertRaises(SpmdTypeError):
            x += y

    # --- out= kwarg ops ---

    def test_out_kwarg_matching_type_ok(self):
        """torch.add(R, R, out=R_tensor): output R matches out tensor R."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self._generate_inputs((4,), self.pg, R)
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_out_kwarg_type_mismatch_rejected(self):
        """torch.add(R, V, out=R_tensor): output V but out is R. Error."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, V)
        c = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.add(a, b, out=c)
        self.assertIn("in-place/out", str(ctx.exception))

    def test_out_kwarg_untyped_ok(self):
        """torch.add(R, R, out=untyped): untyped out tensor is fine."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_out_kwarg_v_output_into_v_ok(self):
        """torch.add(R, V, out=V_tensor): output V, out is V. OK."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, V)
        c = self._generate_inputs((4,), self.pg, V)
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), V)

    def test_inplace_p_p_add_ok(self):
        """P.add_(P) -> P: type preserved (linear op)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), P)


class TestMutateType(unittest.TestCase):
    """Test mutate_type for explicit single-axis type transitions."""

    def test_basic_mutation(self):
        """Mutate a single axis from V to R."""
        x = torch.randn(4)
        assert_type(x, {"dp": V})
        mutate_type(x, "dp", src=V, dst=R)
        self.assertIs(get_axis_local_type(x, "dp"), R)

    def test_shard_sugar_src(self):
        """S(i) is accepted as src and compared as V."""
        x = torch.randn(4)
        assert_type(x, {"dp": S(0)})
        mutate_type(x, "dp", src=S(0), dst=R)
        self.assertIs(get_axis_local_type(x, "dp"), R)

    def test_shard_sugar_dst(self):
        """S(i) is accepted as dst and stored as V."""
        x = torch.randn(4)
        assert_type(x, {"dp": R})
        mutate_type(x, "dp", src=R, dst=S(0))
        self.assertIs(get_axis_local_type(x, "dp"), V)

    def test_preserves_other_axes(self):
        """Mutating one axis must not affect other axes."""
        x = torch.randn(4)
        assert_type(x, {"dp": V, "tp": I})
        mutate_type(x, "dp", src=V, dst=R)
        self.assertIs(get_axis_local_type(x, "dp"), R)
        self.assertIs(get_axis_local_type(x, "tp"), I)

    def test_wrong_src_raises(self):
        """Mismatched src raises SpmdTypeError."""
        x = torch.randn(4)
        assert_type(x, {"dp": V})
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, "dp", src=R, dst=I)

    def test_missing_axis_raises(self):
        """Axis not present in tensor's type raises SpmdTypeError."""
        x = torch.randn(4)
        assert_type(x, {"dp": V})
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, "tp", src=V, dst=R)

    def test_unannotated_tensor_raises(self):
        """Tensor with no type at all raises SpmdTypeError."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, "dp", src=V, dst=R)

    def test_returns_tensor(self):
        """mutate_type returns the tensor for chaining."""
        x = torch.randn(4)
        assert_type(x, {"dp": V})
        result = mutate_type(x, "dp", src=V, dst=R)
        self.assertIs(result, x)


if __name__ == "__main__":
    unittest.main()
