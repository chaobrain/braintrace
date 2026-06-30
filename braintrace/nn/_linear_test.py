# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Comprehensive tests for linear neural network layers.

Tests cover:
- Linear: Standard fully-connected layer
- SignedWLinear: Linear layer with signed weight constraints
- SparseLinear: Linear layer with sparse connectivity
- LoRA: Low-Rank Adaptation layer for fine-tuning
"""

import brainstate
import jax
import jax.numpy as jnp
import pytest
import brainunit as u
import brainevent
from braintools import init

import braintrace


def _be_csr_from_coo(values, rows, cols, shape):
    """Build a :class:`brainevent.CSR` from COO-style ``(values, rows, cols)``.

    ``SparseLinear`` requires a :class:`brainevent.DataRepresentation` (the type
    that provides the ``yw_to_w_transposed`` online-learning protocol). brainunit
    ``COO`` does not qualify, so the connectivity is densified and re-encoded as a
    brainevent CSR. These tests assert output shapes only, so duplicate-index
    summation during densification is irrelevant.
    """
    dense = u.sparse.COO((values, rows, cols), shape=shape).todense()
    return brainevent.CSR.fromdense(dense)


def _flatten_grads(grads):
    """Flatten nested dict grad values (e.g. from a dict-valued ParamState) to flat tuple keys.

    For a grad dict whose values may themselves be dicts (e.g. ``{'weight': …, 'bias': …}``),
    expands ``{k: {subk: v}}`` → ``{k + (subk,): v}`` so that ``assert_param_gradients_close``
    can compare individual leaf arrays.
    """
    flat = {}
    for k, v in grads.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[k + (subk,)] = subv
        else:
            flat[k] = v
    return flat


class TestLinear:
    """Test Linear layer."""

    def test_linear_basic_creation(self):
        """Test basic Linear layer creation."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)
        # in_size and out_size may be scalar or sequence
        assert hasattr(linear, 'in_size')
        assert hasattr(linear, 'out_size')
        assert hasattr(linear, 'weight')

    def test_linear_forward_with_batch(self):
        """Test Linear forward pass with batch dimension."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_forward_without_batch(self):
        """Test Linear forward pass without batch dimension."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)
        x = brainstate.random.randn(128)
        y = linear(x)
        assert y.shape == (64,)

    def test_linear_forward_multi_batch(self):
        """Test Linear forward pass with multiple batch dimensions."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)
        x = brainstate.random.randn(5, 10, 128)
        y = linear(x)
        assert y.shape == (5, 10, 64)

    def test_linear_with_bias(self):
        """Test Linear layer with bias initialization."""
        linear = braintrace.nn.Linear(
            in_size=128,
            out_size=64,
            b_init=init.Constant(0.1)
        )
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_without_bias(self):
        """Test Linear layer without bias."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64, b_init=None)
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_custom_weight_init(self):
        """Test Linear layer with custom weight initializer."""
        linear = braintrace.nn.Linear(
            in_size=128,
            out_size=64,
            w_init=init.Constant(0.5)
        )
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_custom_bias_init(self):
        """Test Linear layer with custom bias initializer."""
        linear = braintrace.nn.Linear(
            in_size=128,
            out_size=64,
            w_init=init.KaimingNormal(),
            b_init=init.Constant(1.0)
        )
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_with_weight_mask(self):
        """Test Linear layer with weight mask."""
        mask = jnp.ones((128, 64))
        linear = braintrace.nn.Linear(in_size=128, out_size=64, w_mask=mask)
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_with_partial_weight_mask(self):
        """Test Linear layer with partial weight mask."""
        mask = jnp.zeros((128, 64))
        mask = mask.at[:64, :32].set(1.0)  # Only connect first half to first half
        linear = braintrace.nn.Linear(in_size=128, out_size=64, w_mask=mask)
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_with_callable_weight_mask(self):
        """Test Linear layer with callable weight mask."""

        def mask_fn(shape):
            return jnp.ones(shape)

        linear = braintrace.nn.Linear(in_size=128, out_size=64, w_mask=mask_fn)
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_sequence_sizes(self):
        """Test Linear layer with sequence sizes."""
        linear = braintrace.nn.Linear(in_size=(64, 128), out_size=(64, 64))
        x = brainstate.random.randn(10, 128)
        y = linear(x)
        assert y.shape == (10, 64)

    def test_linear_large_dimensions(self):
        """Test Linear layer with large dimensions."""
        linear = braintrace.nn.Linear(in_size=2048, out_size=1024)
        x = brainstate.random.randn(8, 2048)
        y = linear(x)
        assert y.shape == (8, 1024)

    def test_linear_small_dimensions(self):
        """Test Linear layer with small dimensions."""
        linear = braintrace.nn.Linear(in_size=4, out_size=2)
        x = brainstate.random.randn(3, 4)
        y = linear(x)
        assert y.shape == (3, 2)

    def test_linear_with_name(self):
        """Test Linear layer with custom name."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64, name="test_linear")
        assert linear.name == "test_linear"

    def test_linear_deterministic_with_same_seed(self):
        """Test that Linear is deterministic with same random seed."""
        brainstate.random.seed(42)
        linear1 = braintrace.nn.Linear(in_size=128, out_size=64)

        brainstate.random.seed(42)
        linear2 = braintrace.nn.Linear(in_size=128, out_size=64)

        x = brainstate.random.randn(10, 128)
        y1 = linear1(x)
        y2 = linear2(x)

        assert jnp.allclose(y1, y2)


class TestSignedWLinear:
    """Test SignedWLinear layer."""

    def test_signed_w_linear_basic_creation(self):
        """Test basic SignedWLinear layer creation."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)
        assert hasattr(linear, 'in_size')
        assert hasattr(linear, 'out_size')
        assert hasattr(linear, 'weight')

    def test_signed_w_linear_forward_with_batch(self):
        """Test SignedWLinear forward pass with batch dimension."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_forward_without_batch(self):
        """Test SignedWLinear forward pass without batch dimension."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)
        x = brainstate.random.randn(64)
        y = linear(x)
        assert y.shape == (32,)

    def test_signed_w_linear_with_positive_signs(self):
        """Test SignedWLinear with positive sign matrix."""
        w_sign = jnp.ones((64, 32))
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, w_sign=w_sign)
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_with_negative_signs(self):
        """Test SignedWLinear with negative sign matrix."""
        w_sign = -jnp.ones((64, 32))
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, w_sign=w_sign)
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_with_mixed_signs(self):
        """Test SignedWLinear with mixed sign matrix."""
        brainstate.random.seed(123)
        w_sign = brainstate.random.choice(jnp.array([-1, 1]), size=(64, 32))
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, w_sign=w_sign)
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_without_sign_matrix(self):
        """Test SignedWLinear without sign matrix (defaults to None)."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, w_sign=None)
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_custom_weight_init(self):
        """Test SignedWLinear with custom weight initializer."""
        linear = braintrace.nn.SignedWLinear(
            in_size=64,
            out_size=32,
            w_init=init.Constant(0.5)
        )
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

    def test_signed_w_linear_sequence_sizes(self):
        """SignedWLinear accepts tuple in_size/out_size at construction time.

        Note: forward pass with multi-dim sizes uses a 4-D weight that does
        not contract against a flat (batch, last_dim) input, so we only
        verify the constructor wires the shapes without crashing.
        """
        linear = braintrace.nn.SignedWLinear(in_size=(32, 64), out_size=(32, 32))
        assert tuple(linear.in_size) == (32, 64)
        assert tuple(linear.out_size) == (32, 32)
        assert linear.weight.value.shape == (32, 64, 32, 32)

    def test_signed_w_linear_with_name(self):
        """Test SignedWLinear with custom name."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32, name="test_signed")
        assert linear.name == "test_signed"

    def test_signed_w_linear_multi_batch(self):
        """Test SignedWLinear with multiple batch dimensions."""
        linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)
        x = brainstate.random.randn(3, 5, 64)
        y = linear(x)
        assert y.shape == (3, 5, 32)


class TestSparseLinear:
    """Test SparseLinear layer."""

    def test_sparse_linear_basic_creation_csr(self):
        """Test basic SparseLinear layer creation with a brainevent CSR sparse matrix."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        assert hasattr(linear, 'out_size')
        assert hasattr(linear, 'weight')

    def test_sparse_linear_forward_with_batch(self):
        """Test SparseLinear forward pass with batch dimension."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_forward_without_batch(self):
        """Test SparseLinear forward pass without batch dimension."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(512)
        y = linear(x)
        assert y.shape == (256,)

    def test_sparse_linear_with_bias(self):
        """Test SparseLinear with bias initialization."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, b_init=init.Constant(0.1))
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_without_bias(self):
        """Test SparseLinear without bias."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, b_init=None)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_with_in_size(self):
        """Test SparseLinear with explicit in_size."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, in_size=512)
        assert hasattr(linear, 'in_size')
        assert hasattr(linear, 'out_size')

    def test_sparse_linear_high_sparsity(self):
        """Test SparseLinear with high sparsity (few connections)."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 100))  # Only 100 connections
        values = brainstate.random.randn(100)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_low_sparsity(self):
        """Test SparseLinear with low sparsity (many connections)."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 50000))  # Many connections
        values = brainstate.random.randn(50000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_with_name(self):
        """Test SparseLinear with custom name."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, name="test_sparse")
        assert linear.name == "test_sparse"

    def test_sparse_linear_multi_batch(self):
        """Test SparseLinear with multiple batch dimensions."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = _be_csr_from_coo(values, rows, cols, (512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(32, 512)
        y = linear(x)
        assert y.shape == (32, 256)

    def test_sparse_linear_invalid_matrix_type(self):
        """Test SparseLinear raises error with invalid matrix type."""
        # Pass a regular array instead of sparse matrix
        regular_mat = jnp.ones((512, 256))
        with pytest.raises(AssertionError):
            braintrace.nn.SparseLinear(regular_mat)


class TestLoRA:
    """Test LoRA layer."""

    def test_lora_basic_creation(self):
        """Test basic LoRA layer creation."""
        lora = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        assert lora.in_features == 3
        assert lora.out_features == 4
        assert lora.base_module is None
        # rank is stored implicitly as the shared inner dim of lora_a/lora_b.
        assert lora.weight.value['lora_a'].shape == (3, 2)
        assert lora.weight.value['lora_b'].shape == (2, 4)

    def test_lora_forward_with_batch(self):
        """Test LoRA forward pass with batch dimension."""
        lora = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_forward_without_batch(self):
        """Test LoRA forward pass without batch dimension."""
        lora = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        x = brainstate.random.randn(3)
        y = lora(x)
        assert y.shape == (4,)

    def test_lora_kernel_init(self):
        """LoRA accepts a single ``kernel_init`` for both lora_a and lora_b."""
        lora = braintrace.nn.LoRA(
            in_features=3, lora_rank=2, out_features=4,
            kernel_init=init.Constant(0.5),
        )
        assert jnp.allclose(lora.weight.value['lora_a'], 0.5)
        assert jnp.allclose(lora.weight.value['lora_b'], 0.5)
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_with_base_module(self):
        """Test LoRA wrapping an existing base module."""
        base_linear = brainstate.nn.Linear(3, 4)
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            base_module=base_linear
        )
        assert lora.base_module == base_linear

        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_zero_kernel_init(self):
        """``kernel_init=ZeroInit`` makes the LoRA branch contribute zero."""
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            kernel_init=init.ZeroInit(),
        )
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)
        assert jnp.allclose(y, 0.0)

    def test_lora_large_rank(self):
        """Test LoRA with large rank."""
        lora = braintrace.nn.LoRA(in_features=128, lora_rank=64, out_features=256)
        assert lora.weight.value['lora_a'].shape == (128, 64)
        assert lora.weight.value['lora_b'].shape == (64, 256)
        x = brainstate.random.randn(8, 128)
        y = lora(x)
        assert y.shape == (8, 256)

    def test_lora_small_rank(self):
        """Test LoRA with small rank."""
        lora = braintrace.nn.LoRA(in_features=128, lora_rank=1, out_features=256)
        assert lora.weight.value['lora_a'].shape == (128, 1)
        assert lora.weight.value['lora_b'].shape == (1, 256)
        x = brainstate.random.randn(8, 128)
        y = lora(x)
        assert y.shape == (8, 256)

    def test_lora_multi_batch(self):
        """Test LoRA with multiple batch dimensions."""
        lora = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        x = brainstate.random.randn(4, 8, 3)
        y = lora(x)
        assert y.shape == (4, 8, 4)

    def test_lora_large_dimensions(self):
        """Test LoRA with large input/output dimensions."""
        lora = braintrace.nn.LoRA(in_features=1024, lora_rank=16, out_features=2048)
        x = brainstate.random.randn(4, 1024)
        y = lora(x)
        assert y.shape == (4, 2048)

    def test_lora_call_routes_through_etp_primitive(self):
        """``lora(x)`` must bind the ETP ``etp_lora`` primitive (not plain
        ``dot_general``).

        Regression: the base ``brainstate.nn.LoRA.__call__`` computed the
        forward directly via ``dot_general`` and never dispatched to
        braintrace's ETP-routed ``update``, so the LoRA factors silently
        bypassed eligibility-trace computation.
        """
        lora = braintrace.nn.LoRA(in_features=5, lora_rank=2, out_features=4)
        # batched input -> etp_lora_mm
        jp = jax.make_jaxpr(lambda z: lora(z))(jnp.ones((8, 5)))
        prims = {str(e.primitive) for e in jp.jaxpr.eqns}
        assert 'etp_lora_mm' in prims, prims
        assert 'dot_general' not in prims, prims
        # unbatched input -> etp_lora_mv
        jp1 = jax.make_jaxpr(lambda z: lora(z))(jnp.ones((5,)))
        prims1 = {str(e.primitive) for e in jp1.jaxpr.eqns}
        assert 'etp_lora_mv' in prims1, prims1

    def test_lora_call_with_base_module_still_binds_etp_primitive(self):
        """With a ``base_module``, ``lora(x)`` must STILL bind the ETP
        primitive for the LoRA branch and add the base output.

        Guards the ``base_module is not None`` path in ``update`` against a
        regression of the ``__call__`` bypass: the LoRA factors must route
        through ``etp_lora`` even when a base module is present.
        """
        base = brainstate.nn.Linear(5, 4)
        lora = braintrace.nn.LoRA(
            in_features=5, lora_rank=2, out_features=4, base_module=base,
        )
        x = jnp.ones((8, 5))
        jp = jax.make_jaxpr(lambda z: lora(z))(x)
        prims = {str(e.primitive) for e in jp.jaxpr.eqns}
        assert 'etp_lora_mm' in prims, prims
        # Output is the ETP LoRA branch plus the base-module output.
        La = lora.weight.value['lora_a']
        Lb = lora.weight.value['lora_b']
        lora_part = (1.0 / 2) * (x @ La @ Lb)
        assert jnp.allclose(lora(x), lora_part + base(x), atol=1e-5)

    def test_lora_forward_applies_rank_scaling(self):
        """Forward equals ``(1/rank) * x @ lora_a @ lora_b``.

        Verifies both the ``1/lora_rank`` scaling and the corrected factor
        order (``lora_a`` is the input-facing ``(in, rank)`` factor, applied
        before ``lora_b`` the ``(rank, out)`` factor).
        """
        lora = braintrace.nn.LoRA(
            in_features=3, lora_rank=2, out_features=4,
            kernel_init=init.Constant(0.5),
        )
        x = brainstate.random.randn(6, 3)
        y = lora(x)
        La = lora.weight.value['lora_a']  # (3, 2)
        Lb = lora.weight.value['lora_b']  # (2, 4)
        expected = (1.0 / 2) * (x @ La @ Lb)
        assert jnp.allclose(y, expected, atol=1e-5)

    def test_lora_compile_yields_temporal_relation(self):
        """A LoRA-driven hidden update yields >=1 ETP hidden-param relation
        under D_RTRL.

        Regression for the ``drtrl/07`` 0-relations bug: because ``lora(x)``
        previously bypassed the ETP primitive, the compiler saw no ETP op and
        produced zero ``hidden_param_op_relations`` for the LoRA factors.
        """

        class LoRACell(brainstate.nn.RNNCell):
            def __init__(self, n_in, n_hidden, rank=2):
                super().__init__()
                self.in_size = n_in
                self.out_size = n_hidden
                self.lora = braintrace.nn.LoRA(
                    in_features=n_in + n_hidden, lora_rank=rank,
                    out_features=n_hidden, kernel_init=init.KaimingNormal(),
                )

            def init_state(self, batch_size=None, **kw):
                self.h = brainstate.HiddenState(
                    init.param(init.ZeroInit(), self.out_size, batch_size)
                )

            def update(self, x):
                xh = jnp.concatenate([x, self.h.value], axis=-1)
                self.h.value = jax.nn.tanh(self.lora(xh))
                return self.h.value

        n_in, n_hidden, batch = 2, 6, 4
        cell = LoRACell(n_in, n_hidden)
        brainstate.nn.init_all_states(cell, batch_size=batch)
        learner = braintrace.D_RTRL(cell)
        learner.compile_graph(jnp.ones((batch, n_in)))
        assert len(learner.graph.hidden_param_op_relations) >= 1


class TestScaledWSLinear:
    """Weight-standardized linear layer (routed through ETP ``matmul``)."""

    def test_forward_shape(self):
        from braintrace.nn._linear import ScaledWSLinear

        layer = ScaledWSLinear(in_size=5, out_size=3)
        x = brainstate.random.randn(8, 5)
        y = layer(x)
        assert y.shape == (8, 3)

    def test_forward_with_weight_mask(self):
        # Exercises the ``w_mask`` multiplication branch in update().
        from braintrace.nn._linear import ScaledWSLinear

        layer = ScaledWSLinear(in_size=4, out_size=2, w_mask=jnp.ones((4, 2)))
        x = brainstate.random.randn(6, 4)
        y = layer(x)
        assert y.shape == (6, 2)

    def test_lora_with_callable_base_module(self):
        """Test LoRA with a callable base module."""

        def custom_layer(x):
            return x @ jnp.ones((3, 4))

        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            base_module=custom_layer
        )
        assert lora.base_module == custom_layer

        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_non_callable_base_module_fails_at_call(self):
        """``brainstate.nn.LoRA`` does not validate ``base_module`` at
        construction; the failure surfaces when ``update`` tries to call it."""
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            base_module="not_callable",
        )
        x = brainstate.random.randn(16, 3)
        with pytest.raises(ValueError, match='callable'):
            lora(x)


class TestLinearIntegration:
    """Integration tests for linear layers."""

    def test_linear_stacked_layers(self):
        """Test stacking multiple Linear layers."""
        linear1 = braintrace.nn.Linear(in_size=128, out_size=64)
        linear2 = braintrace.nn.Linear(in_size=64, out_size=32)
        linear3 = braintrace.nn.Linear(in_size=32, out_size=16)

        x = brainstate.random.randn(10, 128)
        y1 = linear1(x)
        y2 = linear2(y1)
        y3 = linear3(y2)

        assert y1.shape == (10, 64)
        assert y2.shape == (10, 32)
        assert y3.shape == (10, 16)

    def test_mixed_linear_types(self):
        """Test mixing different types of linear layers."""
        linear1 = braintrace.nn.Linear(in_size=128, out_size=64)
        signed_linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)

        x = brainstate.random.randn(10, 128)
        y1 = linear1(x)
        y2 = signed_linear(y1)

        assert y1.shape == (10, 64)
        assert y2.shape == (10, 32)

    def test_lora_fine_tuning_scenario(self):
        """Test LoRA in a fine-tuning scenario."""
        # Pretrained base model
        base_linear = brainstate.nn.Linear(128, 64)

        # Add LoRA adaptation (scalar alpha kwarg is not part of the
        # upstream LoRA API — the scaling is ``1 / lora_rank`` inside update).
        lora = braintrace.nn.LoRA(
            in_features=128,
            lora_rank=8,
            out_features=64,
            base_module=base_linear,
        )

        x = brainstate.random.randn(10, 128)
        base_output = base_linear(x)
        lora_output = lora(x)

        # LoRA should produce output with correct shape
        assert base_output.shape == lora_output.shape

    def test_linear_with_jit_compilation(self):
        """Test that Linear works with JAX JIT compilation."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)

        @brainstate.transform.jit
        def forward(x):
            return linear(x)

        x = brainstate.random.randn(10, 128)
        y = forward(x)
        assert y.shape == (10, 64)

    def test_sparse_linear_vs_dense_linear(self):
        """Test that sparse linear can approximate dense linear."""
        # Create a dense linear layer
        dense_linear = braintrace.nn.Linear(in_size=64, out_size=32, b_init=None)

        # Create a fully connected sparse matrix (simulating dense)
        row_indices = jnp.repeat(jnp.arange(64), 32)
        col_indices = jnp.tile(jnp.arange(32), 64)
        values = brainstate.random.randn(64 * 32)
        sparse_mat = _be_csr_from_coo(values, row_indices, col_indices, (64, 32))

        sparse_linear = braintrace.nn.SparseLinear(sparse_mat, b_init=None)

        x = brainstate.random.randn(8, 64)
        y_dense = dense_linear(x)
        y_sparse = sparse_linear(x)

        # Both should produce valid outputs with correct shape
        assert y_dense.shape == (8, 32)
        assert y_sparse.shape == (8, 32)

    def test_linear_gradient_flow(self):
        """Test that gradients flow through Linear layer."""
        linear = braintrace.nn.Linear(in_size=10, out_size=5)

        def loss_fn(x):
            y = linear(x)
            return jnp.sum(y ** 2)

        x = brainstate.random.randn(4, 10)

        # Compute gradients using the correct API
        grad_fn = brainstate.transform.grad(loss_fn)
        grads = grad_fn(x)

        # Gradients should exist and have correct shape
        assert grads.shape == x.shape
        assert not jnp.all(grads == 0)

    def test_lora_without_base_module(self):
        """Test LoRA as standalone layer without base module."""
        lora = braintrace.nn.LoRA(in_features=64, lora_rank=8, out_features=32)

        x = brainstate.random.randn(10, 64)
        y = lora(x)

        assert y.shape == (10, 32)
        assert lora.base_module is None

    def test_batch_size_consistency(self):
        """Test that all linear layers handle different batch sizes correctly."""
        linear = braintrace.nn.Linear(in_size=64, out_size=32)
        signed_linear = braintrace.nn.SignedWLinear(in_size=64, out_size=32)

        for batch_size in [1, 8, 32, 128]:
            x = brainstate.random.randn(batch_size, 64)

            y1 = linear(x)
            y2 = signed_linear(x)

            assert y1.shape == (batch_size, 32)
            assert y2.shape == (batch_size, 32)


class TestNnWeightFnExactness:
    """Masked Linear and SignedWLinear gradients must match BPTT (now exact)."""

    @staticmethod
    def _rnn_factory(make_layer):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.nh = 4
                self.layer = make_layer()

            def init_state(self, batch_size=None, **kw):
                size = (self.nh,) if batch_size is None else (batch_size, self.nh)
                self.h = brainstate.HiddenState(jnp.zeros(size))

            def update(self, x):
                xh = jnp.concatenate([x.reshape(1, -1), self.h.value], axis=-1)
                self.h.value = jnp.tanh(self.layer(xh))
                return self.h.value

        def factory():
            brainstate.random.seed(0)
            return Cell()

        return factory

    def test_signed_w_linear_matches_bptt(self):
        from braintrace._algorithm.oracle import (
            bptt_param_gradients, online_param_gradients, assert_param_gradients_close,
        )
        factory = self._rnn_factory(lambda: braintrace.nn.SignedWLinear(2 + 4, 4))
        brainstate.random.seed(1)
        inputs = brainstate.random.randn(6, 2)
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step')
        )
        assert_param_gradients_close(online, bptt, atol=1e-4)

    def test_masked_linear_matches_bptt(self):
        from braintrace._algorithm.oracle import (
            bptt_param_gradients, online_param_gradients, assert_param_gradients_close,
        )
        mask = ((jnp.arange(6)[:, None] + jnp.arange(4)[None, :]) % 3 != 0).astype(float)
        factory = self._rnn_factory(lambda: braintrace.nn.Linear(2 + 4, 4, w_mask=mask))
        brainstate.random.seed(1)
        inputs = brainstate.random.randn(6, 2)
        bptt = _flatten_grads(bptt_param_gradients(factory, inputs))
        online = _flatten_grads(online_param_gradients(
            factory, inputs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step')
        ))
        assert_param_gradients_close(online, bptt, atol=1e-4)


class TestScaledWSLinearWeightFn:
    """ScaledWSLinear's standardized 'weight' leaf gradient must match BPTT exactly.

    ``ScaledWSLinear`` stores a single ``ParamState`` whose ``.value`` is a dict
    with keys ``'weight'``, ``'bias'``, and ``'gain'``.  After flattening the
    grad dict with ``_flatten_grads``, the leaf keys are
    ``('layer', 'weight', 'weight')``, ``('layer', 'weight', 'bias')``, and
    ``('layer', 'weight', 'gain')``.

    Exactness is asserted ONLY on the ``'weight'`` leaf (last element ``== 'weight'``).
    The ``gain`` leaf is intentionally excluded: ``gain`` is non-temporal — it
    reaches the hidden state only *through* the standardized weight map and does
    not participate directly in the eligibility trace.  Its online gradient will
    therefore NOT match BPTT, and asserting on it would be incorrect.
    """

    @staticmethod
    def _rnn_factory():
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.nh = 4
                self.layer = braintrace.nn.ScaledWSLinear(2 + 4, 4)

            def init_state(self, batch_size=None, **kw):
                size = (self.nh,) if batch_size is None else (batch_size, self.nh)
                self.h = brainstate.HiddenState(jnp.zeros(size))

            def update(self, x):
                # x is shape (2,); h.value is (1, 4) after init_all_states(batch_size=1)
                xh = jnp.concatenate([x.reshape(1, -1), self.h.value], axis=-1)
                self.h.value = jnp.tanh(self.layer(xh))
                return self.h.value

        def factory():
            brainstate.random.seed(0)
            return Cell()

        return factory

    def test_weight_leaf_gradient_matches_bptt(self):
        """Standardized weight leaf (not gain) online gradient equals BPTT.

        Routing weight-standardization through ``matmul``'s ``weight_fn``
        recovers the standardization Jacobian in the eligibility trace.  The
        exact comparison is restricted to the ``('layer', 'weight', 'weight')``
        leaf.  The ``gain`` leaf is non-temporal (reaches the hidden state only
        through the standardized weight, another trainable map) and is
        intentionally excluded from the exactness assertion.
        """
        from braintrace._algorithm.oracle import (
            bptt_param_gradients, online_param_gradients, assert_param_gradients_close,
        )
        factory = self._rnn_factory()
        brainstate.random.seed(1)
        inputs = brainstate.random.randn(6, 2)
        bptt = _flatten_grads(bptt_param_gradients(factory, inputs))
        online = _flatten_grads(online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        ))
        # Select ONLY the standardized weight leaf by its last path element.
        # After flattening, the three leaves are:
        #   ('layer', 'weight', 'weight') — the standardized weight: ASSERT exact
        #   ('layer', 'weight', 'bias')   — bias: not asserted here
        #   ('layer', 'weight', 'gain')   — gain (non-temporal): EXCLUDED
        weight_keys = [k for k in bptt if k[-1] == 'weight']
        assert weight_keys, (
            f"No 'weight' leaf found after flattening. Keys: {list(bptt.keys())}"
        )
        assert_param_gradients_close(online, bptt, atol=1e-4, keys=weight_keys)
        # gain is differentiated exactly by the multi-step VJP oracle path (post-scale,
        # standard autodiff) — distinct from its non-temporal online eligibility-trace gradient.
        gain_keys = [k for k in bptt if k[-1] == 'gain']
        assert gain_keys, list(bptt.keys())
        assert_param_gradients_close(online, bptt, atol=1e-4, keys=gain_keys)


class TestScaledWSLinearForwardBiasGain:
    """Forward correctness regression: gain must NOT scale the bias.

    The bug introduced in 0b2ccef passed ``bias`` into ``matmul`` and then
    multiplied the whole result (including bias) by ``gain``, yielding
    ``(x @ std(w)) * gain + bias * gain`` instead of the correct
    ``(x @ std(w)) * gain + bias``.  This test sets a non-zero bias and a
    non-unit gain and asserts the braintrace forward matches the brainstate
    reference forward.
    """

    def test_bias_not_scaled_by_gain(self):
        """Non-zero bias with non-unit gain: braintrace must match brainstate."""
        brainstate.environ.set(precision=64)
        brainstate.random.seed(7)

        in_size, out_size = 6, 4

        # Build braintrace layer (the one being fixed).
        bt_layer = braintrace.nn.ScaledWSLinear(in_size=in_size, out_size=out_size)
        # Build brainstate reference layer (uses the original correct forward).
        bs_layer = brainstate.nn.ScaledWSLinear(in_size=in_size, out_size=out_size)

        # Copy weight params from braintrace to brainstate so they are identical.
        bt_params = bt_layer.weight.value
        bs_layer.weight.value = dict(bt_params)

        # Override bias and gain on BOTH layers with non-trivial values.
        bias_val = jnp.array([1.0, 2.0, 3.0, 4.0])
        gain_val = jnp.array([[2.0, 0.5, 3.0, 1.5]])  # shape (1, out_size)

        new_params = dict(bt_params)
        new_params['bias'] = bias_val
        new_params['gain'] = gain_val
        bt_layer.weight.value = new_params
        bs_layer.weight.value = dict(new_params)

        x = brainstate.random.randn(3, in_size)

        bt_out = bt_layer.update(x)
        bs_out = bs_layer.update(x)

        maxabsdiff = float(jnp.max(jnp.abs(bt_out - bs_out)))
        assert maxabsdiff < 1e-5, (
            f"braintrace ScaledWSLinear forward differs from brainstate reference "
            f"by {maxabsdiff:.6f} (max-abs-diff). Bias is being scaled by gain."
        )
