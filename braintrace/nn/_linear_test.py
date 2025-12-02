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

import braintrace
import brainstate
import brainunit as u
import jax.numpy as jnp
import pytest
from braintools import init


class TestLinear:
    """Test Linear layer."""

    def test_linear_basic_creation(self):
        """Test basic Linear layer creation."""
        linear = braintrace.nn.Linear(in_size=128, out_size=64)
        # in_size and out_size may be scalar or sequence
        assert hasattr(linear, 'in_size')
        assert hasattr(linear, 'out_size')
        assert hasattr(linear, 'weight_op')

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
        assert hasattr(linear, 'weight_op')

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
        """Test SignedWLinear with sequence sizes."""
        linear = braintrace.nn.SignedWLinear(in_size=(32, 64), out_size=(32, 32))
        x = brainstate.random.randn(5, 64)
        y = linear(x)
        assert y.shape == (5, 32)

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

    def test_sparse_linear_basic_creation_coo(self):
        """Test basic SparseLinear layer creation with COO sparse matrix."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        assert hasattr(linear, 'out_size')
        assert hasattr(linear, 'weight_op')

    def test_sparse_linear_forward_with_batch(self):
        """Test SparseLinear forward pass with batch dimension."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_forward_without_batch(self):
        """Test SparseLinear forward pass without batch dimension."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(512)
        y = linear(x)
        assert y.shape == (256,)

    def test_sparse_linear_with_bias(self):
        """Test SparseLinear with bias initialization."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, b_init=init.Constant(0.1))
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_without_bias(self):
        """Test SparseLinear without bias."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, b_init=None)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_with_in_size(self):
        """Test SparseLinear with explicit in_size."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, in_size=512)
        assert hasattr(linear, 'in_size')
        assert hasattr(linear, 'out_size')

    def test_sparse_linear_high_sparsity(self):
        """Test SparseLinear with high sparsity (few connections)."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 100))  # Only 100 connections
        values = brainstate.random.randn(100)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_low_sparsity(self):
        """Test SparseLinear with low sparsity (many connections)."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 50000))  # Many connections
        values = brainstate.random.randn(50000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat)
        x = brainstate.random.randn(8, 512)
        y = linear(x)
        assert y.shape == (8, 256)

    def test_sparse_linear_with_name(self):
        """Test SparseLinear with custom name."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

        linear = braintrace.nn.SparseLinear(sparse_mat, name="test_sparse")
        assert linear.name == "test_sparse"

    def test_sparse_linear_multi_batch(self):
        """Test SparseLinear with multiple batch dimensions."""
        brainstate.random.seed(42)
        rows, cols = brainstate.random.randint(0, 512, size=(2, 1000))
        values = brainstate.random.randn(1000)
        sparse_mat = u.sparse.COO((values, rows, cols), shape=(512, 256))

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
        assert lora.lora_rank == 2
        assert lora.out_features == 4
        assert lora.alpha == 1.0
        assert lora.base_module is None

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

    def test_lora_custom_alpha(self):
        """Test LoRA with custom alpha value."""
        lora = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4, alpha=0.5)
        assert lora.alpha == 0.5
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

    def test_lora_custom_b_init(self):
        """Test LoRA with custom B initializer."""
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            B_init=init.Constant(0.1)
        )
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_custom_a_init(self):
        """Test LoRA with custom A initializer."""
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            A_init=init.Constant(0.5)
        )
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_default_b_init_zero(self):
        """Test that LoRA B is initialized to zero by default."""
        lora = braintrace.nn.LoRA(
            in_features=3,
            lora_rank=2,
            out_features=4,
            B_init=init.ZeroInit()
        )
        x = brainstate.random.randn(16, 3)
        y = lora(x)
        assert y.shape == (16, 4)

    def test_lora_large_rank(self):
        """Test LoRA with large rank."""
        lora = braintrace.nn.LoRA(in_features=128, lora_rank=64, out_features=256)
        assert lora.lora_rank == 64
        x = brainstate.random.randn(8, 128)
        y = lora(x)
        assert y.shape == (8, 256)

    def test_lora_small_rank(self):
        """Test LoRA with small rank."""
        lora = braintrace.nn.LoRA(in_features=128, lora_rank=1, out_features=256)
        assert lora.lora_rank == 1
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

    def test_lora_invalid_base_module(self):
        """Test LoRA raises error with invalid base module."""
        with pytest.raises(AssertionError):
            braintrace.nn.LoRA(
                in_features=3,
                lora_rank=2,
                out_features=4,
                base_module="not_callable"
            )


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

        # Add LoRA adaptation
        lora = braintrace.nn.LoRA(
            in_features=128,
            lora_rank=8,
            out_features=64,
            base_module=base_linear,
            alpha=0.1
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
        sparse_mat = u.sparse.COO((values, row_indices, col_indices), shape=(64, 32))

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
