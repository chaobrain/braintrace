# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Bias-gradient correctness for ETP primitives.

Train a tiny recurrent net one step and verify ETP gradients (dW, db)
match BPTT ground-truth.

The merged-ParamState variant uses a single ParamState holding
{'weight': W, 'bias': b}. The test explicitly proves that the bug we
set out to fix (bias gradients being silently zero) is now solved:
D-RTRL produces a db that matches BPTT.
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import braintrace


class TestMMBiasGradient:
    """D-RTRL gradient correctness for etp_mm_p with a bias vector."""

    def _build_cell_merged(self):
        """One-step RNN with weight+bias stored in a merged ParamState."""

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': jnp.ones((4, 4)) * 0.1,
                     'bias': jnp.ones((4,)) * 0.2}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                w = self.p.value['weight']
                b = self.p.value['bias']
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, w, b)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        return cell

    def test_drtrl_grad_matches_bptt_merged_paramstate(self):
        """D-RTRL dW and db must match BPTT for one recurrent step."""
        cell = self._build_cell_merged()
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3
        target = jnp.zeros((1, 4))

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # Extract the dict-valued gradient for cell.p
        # grads_etrace is keyed by path; cell.p is at path ('p',)
        # Its value is a dict {'weight': ..., 'bias': ...}
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ params['weight'] + params['bias'])
            return h.sum()

        bptt = jax.grad(bptt_loss)({'weight': cell.p.value['weight'],
                                    'bias': cell.p.value['bias']})

        npt.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                            err_msg='D-RTRL dW does not match BPTT')
        npt.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                            err_msg='D-RTRL db does not match BPTT (was it zero?)')


class TestMMNonSquareWeight:
    """_mm_yw_to_w must broadcast hidden_dim correctly when in != out.

    The latent bug was ``jnp.expand_dims(hidden_dim, axis=1)`` in the
    gradient-solve context where the batch axis is stripped: for
    ``hidden_dim: (out,)`` and ``trace: (in, out)`` with ``in != out``,
    axis=1 produced shape ``(out, 1)`` and broadcasting failed. Fixed by
    using ``axis=-2`` which inserts the singleton at the correct axis
    in both the batched (trace-update) and unbatched (solve) contexts.
    """

    def test_yw_to_w_non_square_solve_context(self):
        from braintrace._etrace_op.dense import _mm_yw_to_w
        # Gradient-solve shapes (batch axis stripped by outer vmap).
        in_dim, out_dim = 5, 3
        trace_weight = jnp.arange(in_dim * out_dim, dtype=jnp.float32).reshape(
            in_dim, out_dim
        )
        hidden_dim = jnp.array([1.0, 2.0, 3.0])  # (out,)

        out = _mm_yw_to_w(hidden_dim, {'weight': trace_weight}, has_bias=False)

        # Expected: trace[i, o] * hidden_dim[o] — broadcast across in axis.
        expected = trace_weight * hidden_dim[None, :]
        npt.assert_array_equal(out['weight'], expected)
        assert out['weight'].shape == (in_dim, out_dim)

    def test_yw_to_w_non_square_trace_update_context(self):
        from braintrace._etrace_op.dense import _mm_yw_to_w
        # Trace-update shapes (batch retained).
        batch, in_dim, out_dim = 2, 5, 3
        trace_weight = jnp.ones((batch, in_dim, out_dim)) * 0.5
        hidden_dim = jnp.ones((batch, out_dim)) * 2.0

        out = _mm_yw_to_w(hidden_dim, {'weight': trace_weight}, has_bias=False)

        # Expected: trace[b,i,o] * hidden_dim[b,o].
        expected = trace_weight * hidden_dim[:, None, :]
        npt.assert_array_equal(out['weight'], expected)
        assert out['weight'].shape == (batch, in_dim, out_dim)


class TestConvBiasGradient:
    """D-RTRL gradient correctness for etp_conv_p with a bias vector."""

    def test_drtrl_conv_grad_matches_bptt(self):
        """D-RTRL dkernel and dbias must match BPTT for one recurrent step."""
        kernel_init = jnp.ones((3, 4, 4)) * 0.05   # (H, in, out) for 1D conv NHC-HIO
        bias_init = jnp.ones((4,)) * 0.1

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': kernel_init, 'bias': bias_init}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, 6, 4)))

            def update(self, x):
                k = self.p.value['weight']
                b = self.p.value['bias']
                y = braintrace.conv(
                    self.h.value, k, b,
                    strides=(1,), padding='SAME',
                    dimension_numbers=('NHC', 'HIO', 'NHC'),
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 6, 4)))

        x = jnp.ones((1, 6, 4)) * 0.3

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # grads_etrace is keyed by path; cell.p is a dict-valued ParamState
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, 6, 4))
            y = jax.lax.conv_general_dilated(
                lhs=h, rhs=params['weight'],
                window_strides=(1,), padding='SAME',
                dimension_numbers=('NHC', 'HIO', 'NHC'),
            ) + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias': cell.p.value['bias'],
        })

        # Non-zero sanity check for bias gradient
        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        npt.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                            err_msg='D-RTRL dkernel does not match BPTT')
        npt.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                            err_msg='D-RTRL dbias does not match BPTT (was it zero?)')


class TestConv2dBiasGradient:
    """Bias gradient matches BPTT oracle for 2D conv (default NHWC / HWIO layout)."""

    def _run_test(self, spatial_h, spatial_w, kernel_h, kernel_w, in_ch, out_ch,
                  strides, padding, x_val=0.3):
        """Shared helper: build a tiny 2D recurrent cell and compare D-RTRL vs BPTT."""
        # Kernel shape: (Hk, Wk, in_ch, out_ch)  — HWIO layout
        kernel_init = jnp.ones((kernel_h, kernel_w, in_ch, out_ch)) * 0.05
        bias_init = jnp.ones((out_ch,)) * 0.1

        # Compute output spatial size for SAME padding (input size unchanged).
        # For VALID: floor((H - Hk) / stride + 1).
        if padding == 'SAME':
            out_h = -(-spatial_h // strides[0])   # ceil division
            out_w = -(-spatial_w // strides[1])
        else:  # VALID
            out_h = (spatial_h - kernel_h) // strides[0] + 1
            out_w = (spatial_w - kernel_w) // strides[1] + 1

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': kernel_init, 'bias': bias_init}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, out_h, out_w, out_ch)))

            def update(self, x):
                k = self.p.value['weight']
                b = self.p.value['bias']
                y = braintrace.conv(
                    self.h.value, k, b,
                    strides=strides, padding=padding,
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, out_h, out_w, out_ch)))

        x = jnp.ones((1, out_h, out_w, out_ch)) * x_val

        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, out_h, out_w, out_ch))
            y = jax.lax.conv_general_dilated(
                lhs=h, rhs=params['weight'],
                window_strides=strides, padding=padding,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            ) + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias': cell.p.value['bias'],
        })

        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        npt.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                            err_msg='D-RTRL dkernel does not match BPTT')
        npt.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                            err_msg='D-RTRL dbias does not match BPTT (was it zero?)')

    def test_conv2d_default_layout(self):
        """2D conv, default NHWC/HWIO layout, 3x3 kernel, stride 1, SAME padding."""
        self._run_test(
            spatial_h=6, spatial_w=6,
            kernel_h=3, kernel_w=3,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='SAME',
        )

    def test_conv2d_with_valid_padding(self):
        """2D conv, VALID padding, 1x1 kernel — preserves spatial dims without growth.

        A 1x1 kernel with VALID padding is a valid non-trivial test: every output
        position directly maps to one input position (no spatial reduction), so the
        hidden state remains stable in a recurrent cell.  This exercises the VALID
        padding code path and the 2D kernel handling simultaneously.
        """
        self._run_test(
            spatial_h=4, spatial_w=4,
            kernel_h=1, kernel_w=1,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='VALID',
        )

    def test_conv2d_rectangular_kernel(self):
        """2D conv with a non-square 3x1 kernel (horizontal strip)."""
        self._run_test(
            spatial_h=6, spatial_w=6,
            kernel_h=3, kernel_w=1,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='SAME',
        )


class TestSparseMMBiasGradient:
    """D-RTRL gradient correctness for etp_sp_mm_p with a bias vector.

    Uses a hashable stub sparse matrix (backed by a dense 3×4 identity-like
    template) so the test does not depend on saiunit.CSR being hashable in JAX.
    The stub's ``with_data`` reconstructs the dense matrix from the flat data
    vector (one value per non-zero), and ``yw_to_w_transposed`` applies the
    transposed pattern.
    """

    def _make_sparse_mat(self):
        """Return (sparse_mat, dense_template, nnz, dim)."""

        # 4×4 diagonal sparse matrix — 4 non-zeros. Square so that h_new and
        # h_old share the same shape, making the recurrent connection direct.
        dim = 4
        rows = jnp.array([0, 1, 2, 3])
        cols = jnp.array([0, 1, 2, 3])
        dense_template = jnp.eye(dim)  # shape (dim, dim)

        class _HashableStub:
            """Minimal stub satisfying the ETP sparse-mat contract."""

            def __hash__(self):
                return id(self)

            def __eq__(self, other):
                return self is other

            def with_data(self, data):
                # Substitute flat data vector back into the dense template.
                return dense_template.at[rows, cols].set(data)

            def yw_to_w_transposed(self, hidden_dim, trace):
                # Per D-RTRL executor call: trace is (nnz,), hidden_dim is (out,).
                # For a diagonal matrix col_i == i, so:
                #   e^t_i = hidden_dim[col_i] * trace_i = hidden_dim[i] * trace_i.
                if hidden_dim.ndim == 0:
                    return trace * hidden_dim
                n = trace.shape[0]
                return trace * hidden_dim[:n]

        sparse_mat = _HashableStub()
        nnz = dim
        return sparse_mat, dense_template, rows, cols, nnz, dim

    def test_drtrl_sparse_grad_matches_bptt(self):
        """D-RTRL dweight_data and dbias must match BPTT for one recurrent step."""
        sparse_mat, dense_template, rows, cols, nnz, dim = self._make_sparse_mat()

        w_init = jnp.ones((nnz,)) * 0.1
        b_init = jnp.ones((dim,)) * 0.05

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState({'weight': w_init, 'bias': b_init})
                self.h = brainstate.HiddenState(jnp.zeros((1, dim)))

            def update(self, x):
                y = braintrace.sparse_matmul(
                    self.h.value, self.p.value['weight'],
                    sparse_mat=sparse_mat, bias=self.p.value['bias'],
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, dim)))

        x = jnp.ones((1, dim)) * 0.3

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # grads_etrace is keyed by path; cell.p is a dict-valued ParamState
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        # stub.with_data(w) = dense_template.at[rows, cols].set(w)
        def bptt_loss(params):
            h = jnp.zeros((1, dim))
            w_dense = dense_template.at[rows, cols].set(params['weight'])
            y = h @ w_dense + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({'weight': cell.p.value['weight'],
                                    'bias': cell.p.value['bias']})

        # Non-zero sanity check for bias gradient
        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        npt.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                            err_msg='D-RTRL dweight_data does not match BPTT')
        npt.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                            err_msg='D-RTRL dbias does not match BPTT (was it zero?)')
