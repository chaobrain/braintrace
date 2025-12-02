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
Comprehensive tests for RNN neural network cells.

Tests cover:
- ValinaRNNCell: Basic recurrent neural network cell
- GRUCell: Gated Recurrent Unit cell
- MGUCell: Minimal Gated Recurrent Unit cell
- LSTMCell: Long Short-Term Memory cell
- MinimalRNNCell: Minimal RNN cell
- MiniGRU: Minimal GRU cell
- MiniLSTM: Minimal LSTM cell
- LRUCell: Linear Recurrent Unit cell
"""

import pytest

brainstate = pytest.importorskip("brainstate")
braintools = pytest.importorskip("braintools")
brainunit = pytest.importorskip("brainunit")
u = brainunit
jnp = pytest.importorskip("jax.numpy")
init = braintools.init
braintrace = pytest.importorskip("braintrace")


class TestValinaRNNCell:
    """Test ValinaRNNCell."""

    def test_valina_rnn_basic_creation(self):
        """Test basic ValinaRNNCell creation."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'W')
        assert hasattr(cell, 'activation')

    def test_valina_rnn_init_state(self):
        """Test ValinaRNNCell state initialization."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (8, 64)

    def test_valina_rnn_forward_pass(self):
        """Test ValinaRNNCell forward pass."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)
        x = brainstate.random.randn(8, 32)
        h = cell(x)
        assert h.shape == (8, 64)

    def test_valina_rnn_sequential_updates(self):
        """Test ValinaRNNCell with sequential updates."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)

        outputs = []
        for _ in range(5):
            x = brainstate.random.randn(8, 32)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 5
        assert all(o.shape == (8, 64) for o in outputs)

    def test_valina_rnn_reset_state(self):
        """Test ValinaRNNCell state reset."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)

        # Modify state
        cell.h.value = jnp.ones_like(cell.h.value)

        # Reset
        cell.reset_state(batch_size=8)
        assert jnp.allclose(cell.h.value, 0.0)

    def test_valina_rnn_custom_activation(self):
        """Test ValinaRNNCell with custom activation."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64, activation='tanh')
        cell.init_state(batch_size=8)
        x = brainstate.random.randn(8, 32)
        h = cell(x)
        assert h.shape == (8, 64)

    def test_valina_rnn_callable_activation(self):
        """Test ValinaRNNCell with callable activation function."""
        cell = braintrace.nn.ValinaRNNCell(
            in_size=32,
            out_size=64,
            activation=brainstate.nn.relu
        )
        cell.init_state(batch_size=8)
        x = brainstate.random.randn(8, 32)
        h = cell(x)
        assert h.shape == (8, 64)

    def test_valina_rnn_custom_initializers(self):
        """Test ValinaRNNCell with custom initializers."""
        cell = braintrace.nn.ValinaRNNCell(
            in_size=32,
            out_size=64,
            state_init=init.Constant(1.0),
            w_init=init.Constant(0.5),
            b_init=init.Constant(0.1)
        )
        cell.init_state(batch_size=8)
        assert jnp.allclose(cell.h.value, 1.0)

    def test_valina_rnn_with_name(self):
        """Test ValinaRNNCell with custom name."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64, name="test_rnn")
        assert cell.name == "test_rnn"


class TestGRUCell:
    """Test GRUCell."""

    def test_gru_basic_creation(self):
        """Test basic GRUCell creation."""
        cell = braintrace.nn.GRUCell(in_size=128, out_size=256)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'Wz')
        assert hasattr(cell, 'Wr')
        assert hasattr(cell, 'Wh')

    def test_gru_init_state(self):
        """Test GRUCell state initialization."""
        cell = braintrace.nn.GRUCell(in_size=128, out_size=256)
        cell.init_state(batch_size=16)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (16, 256)

    def test_gru_forward_pass(self):
        """Test GRUCell forward pass."""
        cell = braintrace.nn.GRUCell(in_size=128, out_size=256)
        cell.init_state(batch_size=16)
        x = brainstate.random.randn(16, 128)
        h = cell(x)
        assert h.shape == (16, 256)

    def test_gru_sequential_updates(self):
        """Test GRUCell with sequential updates."""
        cell = braintrace.nn.GRUCell(in_size=64, out_size=128)
        cell.init_state(batch_size=8)

        outputs = []
        for _ in range(10):
            x = brainstate.random.randn(8, 64)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 10
        assert all(o.shape == (8, 128) for o in outputs)

    def test_gru_reset_state(self):
        """Test GRUCell state reset."""
        cell = braintrace.nn.GRUCell(in_size=64, out_size=128)
        cell.init_state(batch_size=8)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.reset_state(batch_size=8)
        assert jnp.allclose(cell.h.value, 0.0)

    def test_gru_custom_activation(self):
        """Test GRUCell with custom activation."""
        cell = braintrace.nn.GRUCell(in_size=64, out_size=128, activation='relu')
        cell.init_state(batch_size=8)
        x = brainstate.random.randn(8, 64)
        h = cell(x)
        assert h.shape == (8, 128)

    def test_gru_with_name(self):
        """Test GRUCell with custom name."""
        cell = braintrace.nn.GRUCell(in_size=64, out_size=128, name="test_gru")
        assert cell.name == "test_gru"


class TestMGUCell:
    """Test MGUCell."""

    def test_mgu_basic_creation(self):
        """Test basic MGUCell creation."""
        cell = braintrace.nn.MGUCell(in_size=96, out_size=192)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'Wf')
        assert hasattr(cell, 'Wh')

    def test_mgu_init_state(self):
        """Test MGUCell state initialization."""
        cell = braintrace.nn.MGUCell(in_size=96, out_size=192)
        cell.init_state(batch_size=12)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (12, 192)

    def test_mgu_forward_pass(self):
        """Test MGUCell forward pass."""
        cell = braintrace.nn.MGUCell(in_size=96, out_size=192)
        cell.init_state(batch_size=12)
        x = brainstate.random.randn(12, 96)
        h = cell(x)
        assert h.shape == (12, 192)

    def test_mgu_sequential_updates(self):
        """Test MGUCell with sequential updates."""
        cell = braintrace.nn.MGUCell(in_size=48, out_size=96)
        cell.init_state(batch_size=6)

        outputs = []
        for _ in range(5):
            x = brainstate.random.randn(6, 48)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 5
        assert all(o.shape == (6, 96) for o in outputs)

    def test_mgu_reset_state(self):
        """Test MGUCell state reset."""
        cell = braintrace.nn.MGUCell(in_size=48, out_size=96)
        cell.init_state(batch_size=6)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.reset_state(batch_size=6)
        assert jnp.allclose(cell.h.value, 0.0)


class TestLSTMCell:
    """Test LSTMCell."""

    def test_lstm_basic_creation(self):
        """Test basic LSTMCell creation."""
        cell = braintrace.nn.LSTMCell(in_size=256, out_size=512)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'Wi')
        assert hasattr(cell, 'Wg')
        assert hasattr(cell, 'Wf')
        assert hasattr(cell, 'Wo')

    def test_lstm_init_state(self):
        """Test LSTMCell state initialization."""
        cell = braintrace.nn.LSTMCell(in_size=256, out_size=512)
        cell.init_state(batch_size=20)
        assert hasattr(cell, 'h')
        assert hasattr(cell, 'c')
        assert cell.h.value.shape == (20, 512)
        assert cell.c.value.shape == (20, 512)

    def test_lstm_forward_pass(self):
        """Test LSTMCell forward pass."""
        cell = braintrace.nn.LSTMCell(in_size=256, out_size=512)
        cell.init_state(batch_size=20)
        x = brainstate.random.randn(20, 256)
        h = cell(x)
        assert h.shape == (20, 512)

    def test_lstm_sequential_updates(self):
        """Test LSTMCell with sequential updates."""
        cell = braintrace.nn.LSTMCell(in_size=128, out_size=256)
        cell.init_state(batch_size=10)

        outputs = []
        for _ in range(15):
            x = brainstate.random.randn(10, 128)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 15
        assert all(o.shape == (10, 256) for o in outputs)

    def test_lstm_reset_state(self):
        """Test LSTMCell state reset."""
        cell = braintrace.nn.LSTMCell(in_size=128, out_size=256)
        cell.init_state(batch_size=10)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.c.value = jnp.ones_like(cell.c.value)

        cell.reset_state(batch_size=10)
        assert jnp.allclose(cell.h.value, 0.0)
        assert jnp.allclose(cell.c.value, 0.0)

    def test_lstm_custom_activation(self):
        """Test LSTMCell with custom activation."""
        cell = braintrace.nn.LSTMCell(in_size=64, out_size=128, activation='relu')
        cell.init_state(batch_size=8)
        x = brainstate.random.randn(8, 64)
        h = cell(x)
        assert h.shape == (8, 128)

    def test_lstm_with_name(self):
        """Test LSTMCell with custom name."""
        cell = braintrace.nn.LSTMCell(in_size=64, out_size=128, name="test_lstm")
        assert cell.name == "test_lstm"


class TestMinimalRNNCell:
    """Test MinimalRNNCell."""

    def test_minimal_rnn_basic_creation(self):
        """Test basic MinimalRNNCell creation."""
        cell = braintrace.nn.MinimalRNNCell(in_size=100, out_size=200)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'phi')
        assert hasattr(cell, 'W_u')

    def test_minimal_rnn_init_state(self):
        """Test MinimalRNNCell state initialization."""
        cell = braintrace.nn.MinimalRNNCell(in_size=100, out_size=200)
        cell.init_state(batch_size=24)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (24, 200)

    def test_minimal_rnn_forward_pass(self):
        """Test MinimalRNNCell forward pass."""
        cell = braintrace.nn.MinimalRNNCell(in_size=100, out_size=200)
        cell.init_state(batch_size=24)
        x = brainstate.random.randn(24, 100)
        h = cell(x)
        assert h.shape == (24, 200)

    def test_minimal_rnn_sequential_updates(self):
        """Test MinimalRNNCell with sequential updates."""
        cell = braintrace.nn.MinimalRNNCell(in_size=50, out_size=100)
        cell.init_state(batch_size=12)

        outputs = []
        for _ in range(7):
            x = brainstate.random.randn(12, 50)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 7
        assert all(o.shape == (12, 100) for o in outputs)

    def test_minimal_rnn_reset_state(self):
        """Test MinimalRNNCell state reset."""
        cell = braintrace.nn.MinimalRNNCell(in_size=50, out_size=100)
        cell.init_state(batch_size=12)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.reset_state(batch_size=12)
        assert jnp.allclose(cell.h.value, 0.0)


class TestMiniGRU:
    """Test MiniGRU."""

    def test_minigru_basic_creation(self):
        """Test basic MiniGRU creation."""
        cell = braintrace.nn.MiniGRU(in_size=80, out_size=160)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'W_x')
        assert hasattr(cell, 'W_z')

    def test_minigru_init_state(self):
        """Test MiniGRU state initialization."""
        cell = braintrace.nn.MiniGRU(in_size=80, out_size=160)
        cell.init_state(batch_size=32)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (32, 160)

    def test_minigru_forward_pass(self):
        """Test MiniGRU forward pass."""
        cell = braintrace.nn.MiniGRU(in_size=80, out_size=160)
        cell.init_state(batch_size=32)
        x = brainstate.random.randn(32, 80)
        h = cell(x)
        assert h.shape == (32, 160)

    def test_minigru_sequential_updates(self):
        """Test MiniGRU with sequential updates."""
        cell = braintrace.nn.MiniGRU(in_size=40, out_size=80)
        cell.init_state(batch_size=16)

        outputs = []
        for _ in range(12):
            x = brainstate.random.randn(16, 40)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 12
        assert all(o.shape == (16, 80) for o in outputs)

    def test_minigru_reset_state(self):
        """Test MiniGRU state reset."""
        cell = braintrace.nn.MiniGRU(in_size=40, out_size=80)
        cell.init_state(batch_size=16)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.reset_state(batch_size=16)
        assert jnp.allclose(cell.h.value, 0.0)


class TestMiniLSTM:
    """Test MiniLSTM."""

    def test_minilstm_basic_creation(self):
        """Test basic MiniLSTM creation."""
        cell = braintrace.nn.MiniLSTM(in_size=150, out_size=300)
        assert hasattr(cell, 'in_size')
        assert hasattr(cell, 'out_size')
        assert hasattr(cell, 'W_x')
        assert hasattr(cell, 'W_f')
        assert hasattr(cell, 'W_i')

    def test_minilstm_init_state(self):
        """Test MiniLSTM state initialization."""
        cell = braintrace.nn.MiniLSTM(in_size=150, out_size=300)
        cell.init_state(batch_size=40)
        assert hasattr(cell, 'h')
        assert cell.h.value.shape == (40, 300)

    def test_minilstm_forward_pass(self):
        """Test MiniLSTM forward pass."""
        cell = braintrace.nn.MiniLSTM(in_size=150, out_size=300)
        cell.init_state(batch_size=40)
        x = brainstate.random.randn(40, 150)
        h = cell(x)
        assert h.shape == (40, 300)

    def test_minilstm_sequential_updates(self):
        """Test MiniLSTM with sequential updates."""
        cell = braintrace.nn.MiniLSTM(in_size=75, out_size=150)
        cell.init_state(batch_size=20)

        outputs = []
        for _ in range(10):
            x = brainstate.random.randn(20, 75)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == 10
        assert all(o.shape == (20, 150) for o in outputs)

    def test_minilstm_reset_state(self):
        """Test MiniLSTM state reset."""
        cell = braintrace.nn.MiniLSTM(in_size=75, out_size=150)
        cell.init_state(batch_size=20)

        cell.h.value = jnp.ones_like(cell.h.value)
        cell.reset_state(batch_size=20)
        assert jnp.allclose(cell.h.value, 0.0)


class TestLRUCell:
    """Test LRUCell."""

    def test_lru_basic_creation(self):
        """Test basic LRUCell creation."""
        cell = braintrace.nn.LRUCell(d_model=64, d_hidden=128)
        assert hasattr(cell, 'd_model')
        assert hasattr(cell, 'd_hidden')
        assert hasattr(cell, 'B_re')
        assert hasattr(cell, 'B_im')
        assert hasattr(cell, 'C_re')
        assert hasattr(cell, 'C_im')

    def test_lru_init_state(self):
        """Test LRUCell state initialization."""
        cell = braintrace.nn.LRUCell(d_model=64, d_hidden=128)
        cell.init_state(batch_size=16)
        assert hasattr(cell, 'h_re')
        assert hasattr(cell, 'h_im')
        assert cell.h_re.value.shape == (16, 128)
        assert cell.h_im.value.shape == (16, 128)

    def test_lru_forward_pass(self):
        """Test LRUCell forward pass."""
        cell = braintrace.nn.LRUCell(d_model=64, d_hidden=128)
        cell.init_state(batch_size=16)
        x = brainstate.random.randn(16, 64)
        y = cell(x)
        assert y.shape == (16, 64)

    def test_lru_sequential_updates(self):
        """Test LRUCell with sequential updates."""
        cell = braintrace.nn.LRUCell(d_model=32, d_hidden=64)
        cell.init_state(batch_size=8)

        outputs = []
        for _ in range(20):
            x = brainstate.random.randn(8, 32)
            y = cell(x)
            outputs.append(y)

        assert len(outputs) == 20
        assert all(o.shape == (8, 32) for o in outputs)

    def test_lru_reset_state(self):
        """Test LRUCell state reset."""
        cell = braintrace.nn.LRUCell(d_model=32, d_hidden=64)
        cell.init_state(batch_size=8)

        cell.h_re.value = jnp.ones_like(cell.h_re.value)
        cell.h_im.value = jnp.ones_like(cell.h_im.value)

        cell.reset_state(batch_size=8)
        assert jnp.allclose(cell.h_re.value, 0.0)
        assert jnp.allclose(cell.h_im.value, 0.0)

    def test_lru_custom_parameters(self):
        """Test LRUCell with custom parameters."""
        cell = braintrace.nn.LRUCell(
            d_model=64,
            d_hidden=128,
            r_min=0.1,
            r_max=0.9,
            max_phase=3.14
        )
        assert cell.r_min == 0.1
        assert cell.r_max == 0.9
        assert cell.max_phase == 3.14


class TestRNNCellIntegration:
    """Integration tests for RNN cells."""

    def test_different_batch_sizes(self):
        """Test RNN cells with different batch sizes."""
        cell = braintrace.nn.GRUCell(in_size=32, out_size=64)

        for batch_size in [1, 4, 16, 32]:
            cell.init_state(batch_size=batch_size)
            x = brainstate.random.randn(batch_size, 32)
            h = cell(x)
            assert h.shape == (batch_size, 64)

    def test_sequence_processing(self):
        """Test processing a full sequence through RNN cell."""
        cell = braintrace.nn.LSTMCell(in_size=64, out_size=128)
        cell.init_state(batch_size=8)

        sequence_length = 50
        outputs = []

        for t in range(sequence_length):
            x = brainstate.random.randn(8, 64)
            h = cell(x)
            outputs.append(h)

        assert len(outputs) == sequence_length
        assert all(o.shape == (8, 128) for o in outputs)

    def test_state_persistence(self):
        """Test that state persists across updates."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=4)

        # First update
        x1 = jnp.ones((4, 32))
        h1 = cell(x1)
        state_after_first = cell.h.value.copy()

        # Second update
        x2 = jnp.ones((4, 32))
        h2 = cell(x2)

        # State should have changed
        assert not jnp.allclose(state_after_first, cell.h.value)

    def test_gradient_flow(self):
        """Test that gradients flow through RNN cell."""
        cell = braintrace.nn.GRUCell(in_size=32, out_size=64)
        cell.init_state(batch_size=4)

        def loss_fn(x):
            h = cell(x)
            return jnp.sum(h ** 2)

        x = brainstate.random.randn(4, 32)
        grad_fn = brainstate.transform.grad(loss_fn)
        grads = grad_fn(x)

        assert grads.shape == x.shape
        assert not jnp.all(grads == 0)

    def test_jit_compilation(self):
        """Test RNN cell with JIT compilation."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)

        @brainstate.transform.jit
        def forward(x):
            return cell(x)

        x = brainstate.random.randn(8, 32)
        h = forward(x)
        assert h.shape == (8, 64)

    def test_deterministic_with_seed(self):
        """Test RNN cell determinism with same seed."""
        brainstate.random.seed(42)
        cell1 = braintrace.nn.GRUCell(in_size=32, out_size=64)
        cell1.init_state(batch_size=8)

        brainstate.random.seed(42)
        cell2 = braintrace.nn.GRUCell(in_size=32, out_size=64)
        cell2.init_state(batch_size=8)

        brainstate.random.seed(123)
        x = brainstate.random.randn(8, 32)

        h1 = cell1(x)
        h2 = cell2(x)

        assert jnp.allclose(h1, h2)

    def test_mixed_cell_types(self):
        """Test using different cell types in sequence."""
        gru_cell = braintrace.nn.GRUCell(in_size=64, out_size=128)
        lstm_cell = braintrace.nn.LSTMCell(in_size=128, out_size=256)

        gru_cell.init_state(batch_size=8)
        lstm_cell.init_state(batch_size=8)

        x = brainstate.random.randn(8, 64)
        h1 = gru_cell(x)
        h2 = lstm_cell(h1)

        assert h1.shape == (8, 128)
        assert h2.shape == (8, 256)

    def test_large_dimensions(self):
        """Test RNN cells with large dimensions."""
        cell = braintrace.nn.LSTMCell(in_size=1024, out_size=2048)
        cell.init_state(batch_size=4)

        x = brainstate.random.randn(4, 1024)
        h = cell(x)
        assert h.shape == (4, 2048)

    def test_small_dimensions(self):
        """Test RNN cells with small dimensions."""
        cell = braintrace.nn.GRUCell(in_size=4, out_size=8)
        cell.init_state(batch_size=2)

        x = brainstate.random.randn(2, 4)
        h = cell(x)
        assert h.shape == (2, 8)

    def test_zero_input(self):
        """Test RNN cell with zero input."""
        cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)

        # Initialize with non-zero state
        cell.h.value = jnp.ones_like(cell.h.value)

        # Feed zero input
        x = jnp.zeros((8, 32))
        h = cell(x)

        # Should still produce output
        assert h.shape == (8, 64)

    def test_reset_during_sequence(self):
        """Test resetting state during sequence processing."""
        cell = braintrace.nn.GRUCell(in_size=32, out_size=64)
        cell.init_state(batch_size=8)

        # Process first part of sequence
        for _ in range(10):
            x = brainstate.random.randn(8, 32)
            cell(x)

        state_before_reset = cell.h.value.copy()

        # Reset
        cell.reset_state(batch_size=8)

        # State should be reset to zero
        assert not jnp.allclose(state_before_reset, cell.h.value)
        assert jnp.allclose(cell.h.value, 0.0)

    def test_batch_size_consistency(self):
        """Test consistency across different batch sizes."""
        cells = [
            braintrace.nn.ValinaRNNCell(in_size=32, out_size=64),
            braintrace.nn.GRUCell(in_size=32, out_size=64),
            braintrace.nn.MGUCell(in_size=32, out_size=64),
            braintrace.nn.LSTMCell(in_size=32, out_size=64),
        ]

        for batch_size in [1, 4, 8, 16]:
            for cell in cells:
                cell.init_state(batch_size=batch_size)
                x = brainstate.random.randn(batch_size, 32)
                h = cell(x)
                assert h.shape == (batch_size, 64)
