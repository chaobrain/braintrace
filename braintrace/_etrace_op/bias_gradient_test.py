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
