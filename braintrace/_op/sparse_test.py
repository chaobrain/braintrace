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

"""Tests for the sparse-matmul ETP primitives and the :func:`sparse_matmul` API.

The sparse structure is supplied by the user as a static parameter
(``sparse_mat``), which must be a :class:`brainevent.DataRepresentation`
implementing the methods used by the ETP rules:

* ``with_data(weight_data)`` — substitute new non-zero values into the
  structure, returning a sparse-matmul-able object.
* ``yw_to_w_transposed(hidden_dim, trace)`` — apply the transposed
  pattern when propagating the trace.
* ``yw_to_w(hidden_dim, trace)`` — the non-transposed counterpart.

For end-to-end forward correctness a real :class:`brainevent.CSR` works.
For rule-level and compile/gradient tests a tiny stub *subclassing*
:class:`brainevent.DataRepresentation` fits the contract exactly (and is
hashable, so it can be a static primitive parameter under ``jit``) without
depending on the upstream sparse-matrix surface area.
"""



from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u
import brainevent

import braintrace
from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    etp_sp_mm_p,
    etp_sp_mv_p,
    sparse_matmul,
)

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


class _StubSparseMat(brainevent.DataRepresentation):
    """Minimal :class:`brainevent.DataRepresentation` stub for the rule contract.

    Subclasses :class:`brainevent.DataRepresentation` so it passes the
    ``sparse_matmul`` runtime type check, while staying tiny and independent of
    the upstream sparse-matrix surface area.

    Backed by a dense matrix internally; ``with_data`` rebuilds the dense
    form by reshaping the flat data vector into the original shape.

    ``yw_to_w_transposed`` is called by the D-RTRL executor with
    ``trace`` having the same shape as the weight data (i.e. ``(nnz,)``
    per vmap step). For this stub, all non-zeros are in a specific pattern
    so the transposed application just multiplies each nnz-entry by the
    corresponding column's hidden-dim value.

    To keep rule-level unit tests straightforward (those tests explicitly
    pass the trace in the expected per-executor shape), the stub
    implements the rule correctly for ``trace.ndim == 1``:
    ``result_i = trace_i * hidden_dim[i]`` (for diagonal patterns where
    ``col_i == i``).  When ``trace.ndim == 2`` (legacy test code), it
    falls back to the old broadcast behaviour.
    """

    def __init__(self, dense_template: jnp.ndarray):
        self._template = dense_template
        super().__init__((dense_template,), shape=dense_template.shape)

    def with_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(self._template.shape)

    def yw_to_w_transposed(self, hidden_dim, trace):
        if trace.ndim == 1:
            # Called by the executor per-vmap step: trace is (nnz,),
            # hidden_dim is (out_dim,) or scalar.
            # For this stub's diagonal-like structure we use the first
            # min(nnz, out_dim) elements of hidden_dim.
            n = trace.shape[0]
            if hidden_dim.ndim == 0:
                return trace * hidden_dim
            return trace * hidden_dim[:n]
        # Legacy 2-D trace shape (in, out) — used only in unit tests that
        # pass trace in the old full-matrix form.
        return trace * jnp.expand_dims(hidden_dim, axis=0)

    def yw_to_w(self, hidden_dim, trace):
        # Non-transposed counterpart; the diagonal stub is symmetric so it
        # delegates. Present to complete the DataRepresentation protocol.
        return self.yw_to_w_transposed(hidden_dim, trace)


def _csr_from_dense(dense: jnp.ndarray):
    return brainevent.CSR.fromdense(dense)


# ---------------------------------------------------------------------------
# Forward correctness via real CSR
# ---------------------------------------------------------------------------

class TestForwardWithCSR:

    def test_unbatched_matches_dense(self):
        dense = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        csr = _csr_from_dense(dense)
        x = jnp.array([1.0, 1.0, 1.0])
        out = sparse_matmul(x, csr.data, sparse_mat=csr)
        np.testing.assert_allclose(out, x @ dense)

    def test_batched_matches_dense(self):
        dense = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        csr = _csr_from_dense(dense)
        x = jnp.array([
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
        ])
        out = sparse_matmul(x, csr.data, sparse_mat=csr)
        np.testing.assert_allclose(out, x @ dense)

    def test_with_bias(self):
        dense = jnp.eye(3) * 2.0
        csr = _csr_from_dense(dense)
        x = jnp.ones((4, 3))
        b = jnp.arange(3.0)
        out = sparse_matmul(x, csr.data, sparse_mat=csr, bias=b)
        np.testing.assert_allclose(out, x @ dense + b)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

class _HashableStubMat(_StubSparseMat):
    """Stub that is also ``__hash__``-able so JAX accepts it as a static
    primitive parameter (CSR itself is not hashable in JAX 0.7+)."""

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class TestAutoDispatch:

    def test_unbatched_uses_sp_mv(self):
        stub = _HashableStubMat(jnp.zeros((3, 3)))
        x = jnp.ones(3)
        data = jnp.arange(9.0)
        jaxpr = jax.make_jaxpr(
            lambda x, d: sparse_matmul(x, d, sparse_mat=stub)
        )(x, data)
        assert any(eqn.primitive is etp_sp_mv_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_sp_mm_p for eqn in jaxpr.jaxpr.eqns)

    def test_batched_uses_sp_mm(self):
        stub = _HashableStubMat(jnp.zeros((3, 4)))
        x = jnp.ones((2, 3))
        data = jnp.arange(12.0)
        jaxpr = jax.make_jaxpr(
            lambda x, d: sparse_matmul(x, d, sparse_mat=stub)
        )(x, data)
        assert any(eqn.primitive is etp_sp_mm_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_sp_mv_p for eqn in jaxpr.jaxpr.eqns)


# ---------------------------------------------------------------------------
# brainunit
# ---------------------------------------------------------------------------

class TestBrainunit:

    def test_unitless(self):
        dense = jnp.eye(3) * 2.0
        csr = _csr_from_dense(dense)
        x = jnp.ones(3)
        out = sparse_matmul(x, csr.data, sparse_mat=csr)
        assert not isinstance(out, u.Quantity)


# ---------------------------------------------------------------------------
# ETP rules — yw_to_w / xy_to_dw / init_*
# ---------------------------------------------------------------------------

class TestSpMmEtpRules:

    def test_yw_to_w_via_stub(self):
        rule = ETP_RULES_YW_TO_W[etp_sp_mm_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        # trace is now a dict {'weight': ...}
        trace = {'weight': jnp.ones((3, 4))}
        out = rule(hidden, trace, sparse_mat=stub)
        # stub.yw_to_w_transposed broadcasts hidden over rows.
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])

    def test_yw_to_w_with_bias(self):
        rule = ETP_RULES_YW_TO_W[etp_sp_mm_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace = {'weight': jnp.ones((3, 4)), 'bias': jnp.ones(4)}
        out = rule(hidden, trace, sparse_mat=stub, has_bias=True)
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])
        np.testing.assert_allclose(out['bias'], hidden)

    def test_xy_to_dw_matches_jax_vjp(self):
        rule = ETP_RULES_XY_TO_DW[etp_sp_mm_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        x = jnp.ones((2, 3))
        w_data = jnp.arange(12.0)
        hidden = jnp.ones((2, 4))
        # weights is now a dict
        weights = {'weight': w_data}
        dw = rule(x, hidden, weights, sparse_mat=stub)
        # Equivalent to VJP through `x @ stub.with_data(w)` wrt w.
        _, vjp_fn = jax.vjp(lambda w_: x @ stub.with_data(w_), w_data)
        ref = vjp_fn(hidden)[0]
        np.testing.assert_allclose(dw['weight'], ref)

    def test_xy_to_dw_with_bias(self):
        rule = ETP_RULES_XY_TO_DW[etp_sp_mm_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        x = jnp.ones((2, 3))
        w_data = jnp.arange(12.0)
        b_data = jnp.zeros(4)
        hidden = jnp.ones((2, 4))
        weights = {'weight': w_data, 'bias': b_data}
        dw = rule(x, hidden, weights, sparse_mat=stub, has_bias=True)

        # Bias gradient via VJP: g = hidden (2,4), db = sum(g, axis=0) = (4,).
        # Verify against the JAX VJP reference.
        def _fwd(w_dict):
            return x @ stub.with_data(w_dict['weight']) + w_dict['bias']

        _, vjp_fn = jax.vjp(_fwd, weights)
        ref = vjp_fn(hidden)[0]
        np.testing.assert_allclose(dw['bias'], ref['bias'])

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_sp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        # weight_vars is now a dict
        weight_vars = {'weight': _fake_var((7,))}  # nnz
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (4, 7, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_sp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((7,)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (4, 7, 2)
        assert out['bias'].shape == (4, 5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_sp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((7,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (4, 5, 2)


class TestSpMvEtpRules:

    def test_yw_to_w_via_stub(self):
        rule = ETP_RULES_YW_TO_W[etp_sp_mv_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        # trace is now a dict {'weight': ...}
        trace = {'weight': jnp.ones((3, 4))}
        out = rule(hidden, trace, sparse_mat=stub)
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])

    def test_yw_to_w_with_bias(self):
        rule = ETP_RULES_YW_TO_W[etp_sp_mv_p]
        stub = _StubSparseMat(jnp.zeros((3, 4)))
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace = {'weight': jnp.ones((3, 4)), 'bias': jnp.ones(4)}
        out = rule(hidden, trace, sparse_mat=stub, has_bias=True)
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])
        np.testing.assert_allclose(out['bias'], hidden)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_sp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((7,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (7, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_sp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((7,)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (7, 2)
        assert out['bias'].shape == (5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_sp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((7,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# Bias-gradient correctness (D-RTRL vs BPTT)
# ---------------------------------------------------------------------------

class TestSparseMMBiasGradient:
    """D-RTRL gradient correctness for etp_sp_mm_p with a bias vector.

    Uses a hashable stub sparse matrix (backed by a dense 3×4 identity-like
    template) so the test does not depend on brainunit.CSR being hashable in JAX.
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

        class _HashableStub(brainevent.DataRepresentation):
            """Minimal hashable DataRepresentation satisfying the ETP contract."""

            def __init__(self):
                super().__init__((dense_template,), shape=dense_template.shape)

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

            def yw_to_w(self, hidden_dim, trace):
                return self.yw_to_w_transposed(hidden_dim, trace)

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

        np.testing.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                                   err_msg='D-RTRL dweight_data does not match BPTT')
        np.testing.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                                   err_msg='D-RTRL dbias does not match BPTT (was it zero?)')


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.sparse_matmul is sparse_matmul


# ---------------------------------------------------------------------------
# Runtime type check — sparse_mat must be a brainevent.DataRepresentation
# ---------------------------------------------------------------------------

class TestRuntimeTypeCheck:
    """``sparse_matmul`` rejects anything that is not a brainevent.DataRepresentation."""

    def test_accepts_brainevent_csr(self):
        csr = _csr_from_dense(jnp.eye(3) * 2.0)
        out = sparse_matmul(jnp.ones(3), csr.data, sparse_mat=csr)
        np.testing.assert_allclose(out, jnp.ones(3) @ (jnp.eye(3) * 2.0))

    def test_rejects_brainunit_sparse(self):
        """A brainunit (saiunit) sparse matrix lacks ``yw_to_w_transposed`` and is rejected."""
        bu_csr = u.sparse.CSR.fromdense(jnp.eye(3))
        assert not isinstance(bu_csr, brainevent.DataRepresentation)
        with pytest.raises(TypeError, match='brainevent.DataRepresentation'):
            sparse_matmul(jnp.ones(3), bu_csr.data, sparse_mat=bu_csr)

    def test_rejects_plain_array(self):
        with pytest.raises(TypeError, match='yw_to_w_transposed'):
            sparse_matmul(jnp.ones(3), jnp.ones(3), sparse_mat=jnp.eye(3))

    def test_rejects_duck_typed_non_subclass(self):
        """An object with the right methods but not a DataRepresentation is still rejected."""

        class _DuckMat:
            def with_data(self, data):
                return data

            def yw_to_w_transposed(self, hidden_dim, trace):
                return trace

            def yw_to_w(self, hidden_dim, trace):
                return trace

        with pytest.raises(TypeError, match='brainevent.DataRepresentation'):
            sparse_matmul(jnp.ones(3), jnp.ones(3), sparse_mat=_DuckMat())


# ---------------------------------------------------------------------------
# weight_fn / bias_fn transforms
# ---------------------------------------------------------------------------

class TestSparseWeightFnBiasFn:
    """Tests for weight_fn / bias_fn support in sparse_matmul."""

    def test_forward_applies_weight_fn(self):
        """weight_fn=lambda w: w**2 should be applied to the nnz data before matmul."""
        dense = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        csr = _csr_from_dense(dense)
        x = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        out = sparse_matmul(x, csr.data, sparse_mat=csr, weight_fn=lambda w: w ** 2)
        # Reference: apply weight_fn to nnz data, reconstruct, and matmul
        ref_dense = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 9.0],
        ])
        ref = x @ ref_dense
        np.testing.assert_allclose(u.get_mantissa(out), u.get_mantissa(ref), atol=1e-4)

    def test_forward_applies_bias_fn(self):
        """bias_fn=lambda b: b*2 should be applied to the bias before adding."""
        dense = jnp.eye(3) * 2.0
        csr = _csr_from_dense(dense)
        x = jnp.ones((2, 3))
        b = jnp.arange(3.0)
        out = sparse_matmul(x, csr.data, sparse_mat=csr, bias=b, bias_fn=lambda b: b * 2.0)
        ref = x @ dense + b * 2.0
        np.testing.assert_allclose(u.get_mantissa(out), u.get_mantissa(ref), atol=1e-4)

    def test_forward_no_fns_unchanged(self):
        """Passing weight_fn=None, bias_fn=None must be bit-identical to no-fn baseline."""
        dense = jnp.eye(3) * 2.0
        csr = _csr_from_dense(dense)
        x = jnp.ones((2, 3))
        b = jnp.arange(3.0)
        out_with_none = sparse_matmul(x, csr.data, sparse_mat=csr, bias=b,
                                      weight_fn=None, bias_fn=None)
        out_baseline = sparse_matmul(x, csr.data, sparse_mat=csr, bias=b)
        np.testing.assert_allclose(
            u.get_mantissa(out_with_none),
            u.get_mantissa(out_baseline),
            atol=0,
        )

    def test_xy_to_dw_matches_vjp_through_weight_fn(self):
        """xy_to_dw rule must match jax.vjp through weight_fn(w)**2."""
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_sp_mm_p
        from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp

        # Use the hashable stub so it can be a static primitive param
        stub = _HashableStubMat(jnp.zeros((3, 4)))
        x = jnp.ones((2, 3))
        w_data = jnp.arange(1.0, 13.0)  # shape (12,)
        hidden = brainstate.random.randn(2, 4)

        rule = ETP_RULES_XY_TO_DW[etp_sp_mm_p]
        params = {
            'sparse_mat': stub,
            'has_bias': False,
            'weight_fn': lambda w: w ** 2,
            'bias_fn': None,
        }
        weights = {'weight': w_data}

        def impl(wd):
            return x @ stub.with_data(wd['weight'] ** 2)

        assert_xy_to_dw_matches_vjp(
            rule=rule, impl=impl, x=x, hidden_dim=hidden,
            weights=weights, params=params, atol=1e-4,
        )

    def test_xy_to_dw_matches_vjp_with_bias_fn(self):
        """xy_to_dw rule must match jax.vjp through both weight_fn and bias_fn."""
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_sp_mm_p
        from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp

        stub = _HashableStubMat(jnp.zeros((3, 4)))
        x = jnp.ones((2, 3))
        w_data = jnp.arange(1.0, 13.0)
        b_data = jnp.ones(4) * 0.5
        hidden = brainstate.random.randn(2, 4)

        rule = ETP_RULES_XY_TO_DW[etp_sp_mm_p]
        params = {
            'sparse_mat': stub,
            'has_bias': True,
            'weight_fn': lambda w: w ** 2,
            'bias_fn': lambda b: b * 3.0,
        }
        weights = {'weight': w_data, 'bias': b_data}

        def impl(wd):
            return x @ stub.with_data(wd['weight'] ** 2) + wd['bias'] * 3.0

        assert_xy_to_dw_matches_vjp(
            rule=rule, impl=impl, x=x, hidden_dim=hidden,
            weights=weights, params=params, atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Audit C3/H1 regression: stock (unwrapped) brainevent.CSR usability +
# batched D-RTRL trace recurrence.
# ---------------------------------------------------------------------------

class TestStockCSRHashable:
    """H1: a stock ``brainevent.CSR`` — exactly as the module docstring shows,
    with no user-side wrapper — must work as ``sparse_mat`` under
    ``jax.jit``/``jax.grad``.

    ``brainevent.CSR`` (like every ``DataRepresentation``) defines ``__eq__``
    without a matching ``__hash__``, so binding it directly as a jaxpr
    equation parameter previously raised ``TypeError: parameters to jaxpr
    equations must have __hash__`` under JAX >= 0.7 -- the documented usage
    failed outright.
    """

    def test_stock_csr_runs_under_jit_and_grad(self):
        with brainstate.environ.context(precision=64):
            mask = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ], dtype=bool)
            csr = brainevent.CSR.fromdense(jnp.asarray(mask, dtype=float))
            x = jnp.ones((2, 3))

            @jax.jit
            def f(w):
                return braintrace.sparse_matmul(x, w, sparse_mat=csr).sum()

            out = f(csr.data)
            assert bool(jnp.isfinite(out))

            g = jax.grad(f)(csr.data)
            assert g.shape == csr.data.shape
            assert bool(jnp.all(jnp.isfinite(g)))


class TestBatchedSparseDRTRLOracle:
    """C3: ``_sp_mm_yw_to_w`` (the ``etp_sp_mm_p`` D-RTRL trace-recurrence rule)
    is handed a leading batch axis -- ``hidden_dim: (batch, out)``,
    ``trace['weight']: (batch, nnz)`` -- straight from the online-trace update.
    ``brainevent``'s ``yw_to_w_transposed`` kernel only accepts 1-D operands, so
    the rule must ``jax.vmap`` internally rather than handing it the batched
    arrays directly. Exactly-diagonal recurrence: D-RTRL (an *exact* algorithm)
    must match the BPTT oracle element-wise for a batch > 1 model.
    """

    def test_batched_drtrl_matches_bptt(self):
        from braintrace._algorithm.oracle import (
            bptt_param_gradients,
            online_param_gradients_singlestep_naive,
        )

        with brainstate.environ.context(precision=64):
            leak = 0.5
            n_rec = 4
            batch = 2
            dense_mask = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ], dtype=bool)
            csr = brainevent.CSR.fromdense(jnp.asarray(dense_mask, dtype=jnp.float64))
            n_in = dense_mask.shape[0]
            nnz = csr.data.shape[0]

            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = brainstate.ParamState(0.1 * brainstate.random.randn(nnz))
                    self.h = brainstate.HiddenState(jnp.zeros((batch, n_rec)))

                def update(self, x):
                    drive = braintrace.sparse_matmul(x, self.w.value, sparse_mat=csr)
                    self.h.value = leak * self.h.value + drive
                    return self.h.value

            def factory():
                # Deterministic weight init: bptt_param_gradients and
                # online_param_gradients_singlestep_naive each call ``factory()``
                # independently, so re-seeding here (rather than relying on
                # ambient RNG state) is what makes the two models start from
                # the *same* weights -- otherwise the "gradient mismatch" would
                # just be two different random points, not a genuine algorithm
                # discrepancy.
                brainstate.random.seed(0)
                return Net()

            xs = 0.3 * brainstate.random.randn(3, batch, n_in)

            g_bptt = bptt_param_gradients(factory, xs)
            g_online = online_param_gradients_singlestep_naive(
                factory, xs,
                algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
            )

            assert set(g_bptt.keys()) == set(g_online.keys())
            for key in g_bptt:
                a = jnp.asarray(g_bptt[key])
                b = jnp.asarray(g_online[key])
                assert a.shape == b.shape
                denom = max(float(jnp.abs(a).max()), 1e-12)
                rel = float(jnp.abs(a - b).max() / denom)
                assert rel < 1e-10, f'{key}: max_rel_err={rel:.3e}'


# ---------------------------------------------------------------------------
# Audit Task 11 (T3): first-principles ``yw_to_w`` from ``jax.jacobian``
# ---------------------------------------------------------------------------

def _sparse_row_col_of_nnz(csr):
    """Derive per-nnz (row, col) indices directly from CSR structure.

    Built independently of ``yw_to_w_transposed`` so it can serve as ground
    truth: ``csr.indices`` already *is* the per-nnz column index (standard
    CSR layout), and the per-nnz row index is recovered from ``indptr`` by
    repeating each row index by its nnz count.
    """
    indptr = np.asarray(csr.indptr)
    row_of_nnz = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
    col_of_nnz = np.asarray(csr.indices)
    return row_of_nnz, col_of_nnz


def _random_masked_csr(mask: np.ndarray, seed: int):
    brainstate.random.seed(seed)
    values = brainstate.random.randn(*mask.shape)
    dense = jnp.where(jnp.asarray(mask), values, 0.0)
    return brainevent.CSR.fromdense(dense)


class TestYwToWFirstPrinciplesFromJacobian:
    """Derive ``_sp_mv_yw_to_w`` / ``_sp_mm_yw_to_w`` from ``jax.jacobian`` of
    the primitive's own forward on a real, non-diagonal :class:`brainevent.CSR`.

    For ``y = x @ W`` with ``W`` sparse, ``partial y_o / partial data_k`` is
    nonzero only for the nnz entries ``k`` whose column equals ``o``, in
    which case it equals ``x`` at that entry's row — a structural fact
    confirmed here numerically against ``row_of_nnz`` / ``col_of_nnz``
    derived directly from the CSR's own ``indptr`` / ``indices`` (never from
    ``yw_to_w_transposed``). Because the Jacobian is "diagonal" in this
    per-nnz-column sense, the general contraction of a cotangent against it
    reduces to indexing the cotangent by each nnz's column and multiplying
    into the trace — exactly what the production rule computes. Checked
    with random (never all-ones) cotangents and traces.
    """

    @staticmethod
    def _mask():
        return np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ], dtype=bool)  # (in=3, out=4), deliberately non-diagonal/non-square

    def test_mv_yw_to_w_matches_jacobian_contraction(self):
        from braintrace._op.sparse import _sp_mv_yw_to_w
        csr = self._random_masked_csr_mv()
        n_in, n_out = csr.shape
        row_of_nnz, col_of_nnz = _sparse_row_col_of_nnz(csr)
        nnz = csr.data.shape[0]

        brainstate.random.seed(511)
        x0 = brainstate.random.randn(n_in)

        def fwd(w):
            return x0 @ csr.with_data(w)

        J = jax.jacobian(fwd)(csr.data)  # (out, nnz)
        for o in range(n_out):
            for k in range(nnz):
                expected = x0[row_of_nnz[k]] if col_of_nnz[k] == o else 0.0
                np.testing.assert_allclose(
                    J[o, k], expected, atol=1e-10,
                    err_msg=f'Jacobian mismatch at o={o}, k={k}',
                )

        g = brainstate.random.randn(n_out)
        trace_w = brainstate.random.randn(nnz)
        trace_b = brainstate.random.randn(n_out)

        # General contraction, specialized by the proven per-column-diagonal
        # structure: each nnz only "sees" the cotangent component at its own
        # column. Built from `col_of_nnz` (raw CSR structure), never from
        # the rule's own `yw_to_w_transposed` call.
        ref_w = trace_w * g[col_of_nnz]
        ref_b = trace_b * g

        out = _sp_mv_yw_to_w(
            g, {'weight': trace_w, 'bias': trace_b}, sparse_mat=csr, has_bias=True,
        )
        np.testing.assert_allclose(out['weight'], ref_w, atol=1e-10)
        np.testing.assert_allclose(out['bias'], ref_b, atol=1e-10)

    def test_mm_yw_to_w_matches_jacobian_contraction(self):
        from braintrace._op.sparse import _sp_mm_yw_to_w
        csr = self._random_masked_csr_mm()
        n_in, n_out = csr.shape
        row_of_nnz, col_of_nnz = _sparse_row_col_of_nnz(csr)
        nnz = csr.data.shape[0]
        batch = 2

        brainstate.random.seed(512)
        x0 = brainstate.random.randn(batch, n_in)

        def fwd(w):
            return x0 @ csr.with_data(w)

        J = jax.jacobian(fwd)(csr.data)  # (batch, out, nnz)
        for b in range(batch):
            for o in range(n_out):
                for k in range(nnz):
                    expected = x0[b, row_of_nnz[k]] if col_of_nnz[k] == o else 0.0
                    np.testing.assert_allclose(
                        J[b, o, k], expected, atol=1e-10,
                        err_msg=f'Jacobian mismatch at b={b}, o={o}, k={k}',
                    )

        g = brainstate.random.randn(batch, n_out)
        trace_w = brainstate.random.randn(batch, nnz)
        trace_b = brainstate.random.randn(batch, n_out)

        ref_w = trace_w * g[:, col_of_nnz]
        ref_b = trace_b * g

        out = _sp_mm_yw_to_w(
            g, {'weight': trace_w, 'bias': trace_b}, sparse_mat=csr, has_bias=True,
        )
        np.testing.assert_allclose(out['weight'], ref_w, atol=1e-10)
        np.testing.assert_allclose(out['bias'], ref_b, atol=1e-10)

    def _random_masked_csr_mv(self):
        return _random_masked_csr(self._mask(), seed=513)

    def _random_masked_csr_mm(self):
        return _random_masked_csr(self._mask(), seed=514)
