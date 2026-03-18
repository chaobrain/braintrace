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
Comprehensive tests for readout neural network layers.

Tests cover:
- LeakyRateReadout: Leaky integration mechanism for continuous input signals
"""

import brainstate
import brainunit as u
import jax.numpy as jnp
from braintools import init

import braintrace

# Set default dt for all tests
brainstate.environ.set(dt=0.1 * u.ms)


class TestLeakyRateReadout:
    """Test LeakyRateReadout layer."""

    def test_leaky_rate_readout_basic_creation(self):
        """Test basic LeakyRateReadout layer creation."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        assert hasattr(readout, 'in_size')
        assert hasattr(readout, 'out_size')
        assert hasattr(readout, 'tau')
        assert hasattr(readout, 'decay')
        assert hasattr(readout, 'weight_op')

    def test_leaky_rate_readout_default_tau(self):
        """Test LeakyRateReadout with default tau value."""
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        assert readout.tau is not None
        assert readout.decay is not None

    def test_leaky_rate_readout_custom_tau(self):
        """Test LeakyRateReadout with custom tau value."""
        tau_value = 10.0 * u.ms
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10, tau=tau_value)
        assert readout.tau is not None

    def test_leaky_rate_readout_init_state_with_batch(self):
        """Test LeakyRateReadout state initialization with batch size."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        readout.init_state(batch_size=32)
        assert hasattr(readout, 'r')
        assert readout.r.value.shape == (32, 10)

    def test_leaky_rate_readout_init_state_without_batch(self):
        """Test LeakyRateReadout state initialization without batch size."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        readout.init_state(batch_size=None)
        assert hasattr(readout, 'r')
        assert readout.r.value.shape == (10,)

    def test_leaky_rate_readout_reset_state(self):
        """Test LeakyRateReadout state reset."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        readout.init_state(batch_size=32)

        # Modify state
        readout.r.value = jnp.ones_like(readout.r.value)

        # Reset state
        readout.reset_state(batch_size=32)
        assert u.math.allclose(readout.r.value, 0.0)

    def test_leaky_rate_readout_forward_with_batch(self):
        """Test LeakyRateReadout forward pass with batch dimension."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        readout.init_state(batch_size=32)

        x = brainstate.random.randn(32, 256)
        output = readout.update(x)
        assert output.shape == (32, 10)

    def test_leaky_rate_readout_forward_without_batch(self):
        """Test LeakyRateReadout forward pass without batch dimension."""
        readout = braintrace.nn.LeakyRateReadout(in_size=256, out_size=10)
        readout.init_state(batch_size=None)

        x = brainstate.random.randn(256)
        output = readout.update(x)
        assert output.shape == (10,)

    def test_leaky_rate_readout_sequential_updates(self):
        """Test LeakyRateReadout with sequential updates."""
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        readout.init_state(batch_size=16)

        outputs = []
        for _ in range(5):
            x = brainstate.random.randn(16, 128)
            output = readout.update(x)
            outputs.append(output)

        assert len(outputs) == 5
        assert all(o.shape == (16, 10) for o in outputs)

    def test_leaky_rate_readout_state_accumulation(self):
        """Test that LeakyRateReadout accumulates state over time."""
        readout = braintrace.nn.LeakyRateReadout(in_size=64, out_size=8, tau=10.0 * u.ms)
        readout.init_state(batch_size=4)

        # First update
        x1 = jnp.ones((4, 64))
        output1 = readout.update(x1)

        # Second update - state should include contribution from first update
        x2 = jnp.ones((4, 64))
        output2 = readout.update(x2)

        # Second output should be different from first (due to accumulated state)
        assert not u.math.allclose(output1, output2)

    def test_leaky_rate_readout_custom_w_init(self):
        """Test LeakyRateReadout with custom weight initializer."""
        readout = braintrace.nn.LeakyRateReadout(
            in_size=128,
            out_size=10,
            w_init=init.Constant(0.5)
        )
        readout.init_state(batch_size=16)

        x = brainstate.random.randn(16, 128)
        output = readout.update(x)
        assert output.shape == (16, 10)

    def test_leaky_rate_readout_custom_r_init(self):
        """Test LeakyRateReadout with custom state initializer."""
        readout = braintrace.nn.LeakyRateReadout(
            in_size=128,
            out_size=10,
            r_init=init.Constant(1.0)
        )
        readout.init_state(batch_size=16)

        # State should be initialized to 1.0
        assert u.math.allclose(readout.r.value, 1.0)

    def test_leaky_rate_readout_with_name(self):
        """Test LeakyRateReadout with custom name."""
        readout = braintrace.nn.LeakyRateReadout(
            in_size=256,
            out_size=10,
            name="test_readout"
        )
        assert readout.name == "test_readout"

    def test_leaky_rate_readout_large_dimensions(self):
        """Test LeakyRateReadout with large dimensions."""
        readout = braintrace.nn.LeakyRateReadout(in_size=2048, out_size=512)
        readout.init_state(batch_size=8)

        x = brainstate.random.randn(8, 2048)
        output = readout.update(x)
        assert output.shape == (8, 512)

    def test_leaky_rate_readout_small_dimensions(self):
        """Test LeakyRateReadout with small dimensions."""
        readout = braintrace.nn.LeakyRateReadout(in_size=4, out_size=2)
        readout.init_state(batch_size=2)

        x = brainstate.random.randn(2, 4)
        output = readout.update(x)
        assert output.shape == (2, 2)

    def test_leaky_rate_readout_different_batch_sizes(self):
        """Test LeakyRateReadout with different batch sizes."""
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)

        for batch_size in [1, 8, 32, 64]:
            readout.init_state(batch_size=batch_size)
            x = brainstate.random.randn(batch_size, 128)
            output = readout.update(x)
            assert output.shape == (batch_size, 10)

    def test_leaky_rate_readout_decay_computation(self):
        """Test that decay is computed correctly from tau."""
        tau_value = 5.0 * u.ms
        readout = braintrace.nn.LeakyRateReadout(
            in_size=128,
            out_size=10,
            tau=tau_value
        )

        # Decay should be exp(-1/tau_normalized) where tau_normalized = tau/dt
        tau_normalized = u.maybe_decimal(readout.tau / brainstate.environ.get_dt())
        expected_decay = u.math.exp(-1.0 / tau_normalized)
        assert u.math.allclose(readout.decay, expected_decay)

    def test_leaky_rate_readout_zero_input(self):
        """Test LeakyRateReadout with zero input."""
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        readout.init_state(batch_size=16)

        # Initialize with non-zero state
        readout.r.value = jnp.ones_like(readout.r.value)
        initial_state = readout.r.value.copy()

        # Feed zero input
        x = jnp.zeros((16, 128))
        output = readout.update(x)

        # State should decay toward zero
        assert jnp.all(jnp.abs(output) < jnp.abs(initial_state))

    def test_leaky_rate_readout_deterministic_with_seed(self):
        """Test that LeakyRateReadout is deterministic with same random seed."""
        brainstate.random.seed(42)
        readout1 = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        readout1.init_state(batch_size=16)

        brainstate.random.seed(42)
        readout2 = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        readout2.init_state(batch_size=16)

        brainstate.random.seed(123)
        x = brainstate.random.randn(16, 128)

        output1 = readout1.update(x)
        output2 = readout2.update(x)

        assert u.math.allclose(output1, output2)


class TestReadoutIntegration:
    """Integration tests for readout layers."""

    def test_multiple_rate_readouts_stacked(self):
        """Test stacking multiple rate readout layers."""
        readout1 = braintrace.nn.LeakyRateReadout(in_size=256, out_size=128)
        readout2 = braintrace.nn.LeakyRateReadout(in_size=128, out_size=64)
        readout3 = braintrace.nn.LeakyRateReadout(in_size=64, out_size=10)

        readout1.init_state(batch_size=16)
        readout2.init_state(batch_size=16)
        readout3.init_state(batch_size=16)

        x = brainstate.random.randn(16, 256)
        y1 = readout1.update(x)
        y2 = readout2.update(y1)
        y3 = readout3.update(y2)

        assert y1.shape == (16, 128)
        assert y2.shape == (16, 64)
        assert y3.shape == (16, 10)

    def test_rate_readout_temporal_dynamics(self):
        """Test temporal dynamics of rate readout over multiple steps."""
        readout = braintrace.nn.LeakyRateReadout(
            in_size=64,
            out_size=10,
            tau=10.0 * u.ms
        )
        readout.init_state(batch_size=8)

        outputs = []
        for t in range(20):
            # Constant input
            x = jnp.ones((8, 64))
            output = readout.update(x)
            outputs.append(output)

        # Output should converge over time
        assert len(outputs) == 20

        # Later outputs should be more similar to each other than early outputs
        early_diff = jnp.mean(jnp.abs(outputs[1] - outputs[0]))
        late_diff = jnp.mean(jnp.abs(outputs[-1] - outputs[-2]))
        assert late_diff < early_diff

    def test_readout_with_jit_compilation(self):
        """Test that readout layers work with JAX JIT compilation."""
        readout = braintrace.nn.LeakyRateReadout(in_size=128, out_size=10)
        readout.init_state(batch_size=16)

        @brainstate.transform.jit
        def forward(x):
            return readout.update(x)

        x = brainstate.random.randn(16, 128)
        y = forward(x)
        assert y.shape == (16, 10)

    def test_rate_readout_gradient_flow(self):
        """Test that gradients flow through rate readout layer."""
        readout = braintrace.nn.LeakyRateReadout(in_size=64, out_size=10)
        readout.init_state(batch_size=8)

        def loss_fn(x):
            y = readout.update(x)
            return jnp.sum(y ** 2)

        x = brainstate.random.randn(8, 64)

        grad_fn = brainstate.transform.grad(loss_fn)
        grads = grad_fn(x)

        # Gradients should exist and have correct shape
        assert grads.shape == x.shape
        assert not jnp.all(grads == 0)

    def test_rate_readout_state_persistence(self):
        """Test that rate readout state persists across updates."""
        readout = braintrace.nn.LeakyRateReadout(in_size=32, out_size=8)
        readout.init_state(batch_size=4)

        # First update
        x1 = jnp.ones((4, 32))
        output1 = readout.update(x1)
        state_after_first = readout.r.value.copy()

        # Second update with zero input
        x2 = jnp.zeros((4, 32))
        output2 = readout.update(x2)

        # State should have decayed but not reset to zero
        assert not u.math.allclose(readout.r.value, 0.0)
        assert not u.math.allclose(readout.r.value, state_after_first)

    def test_rate_readout_reset_vs_reinit(self):
        """Test that reset_state and init_state produce same initial state."""
        readout = braintrace.nn.LeakyRateReadout(in_size=64, out_size=10)

        # Initialize
        readout.init_state(batch_size=16)
        state_after_init = readout.r.value.copy()

        # Modify state
        readout.r.value = jnp.ones_like(readout.r.value)

        # Reset
        readout.reset_state(batch_size=16)
        state_after_reset = readout.r.value.copy()

        # Both should be zero (default initialization)
        assert u.math.allclose(state_after_init, state_after_reset)
