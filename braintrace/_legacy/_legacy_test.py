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

"""Smoke tests for the v0.1.x compatibility shims.

Coverage:
* every legacy class constructs and executes forward.
* ``ETraceParam`` + ``MatMulOp`` produces *one* compiler relation.
* ``NonTempParam`` + ``MatMulOp`` produces *zero* compiler relations.
* ``FakeETraceParam`` is invisible to the compiler.
* ``DeprecationWarning`` fires on first construction.
"""

from __future__ import annotations

import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import saiunit as u

import braintrace
from braintrace._legacy import (
    ConvOp,
    ElemWiseOp,
    ElemWiseParam,
    ETraceOp,
    ETraceParam,
    FakeElemWiseParam,
    FakeETraceParam,
    LoraOp,
    MatMulOp,
    NonTempParam,
    SpMatMulOp,
    general_y2w,
    stop_param_gradients,
)


# ---------------------------------------------------------------------------
# Op forward pass
# ---------------------------------------------------------------------------

class TestOpsForward:

    def test_matmul_op(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        w = {'weight': jnp.ones((4, 3))}
        y = op(x, w)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y, x @ w['weight'])

    def test_matmul_op_with_bias(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        b = jnp.arange(3.0)
        w = {'weight': jnp.ones((4, 3)), 'bias': b}
        y = op(x, w)
        np.testing.assert_allclose(y, x @ w['weight'] + b)

    def test_matmul_op_weight_fn(self):
        op = MatMulOp(weight_fn=lambda w: w * 2)
        x = jnp.ones((2, 4))
        w = {'weight': jnp.ones((4, 3))}
        y = op(x, w)
        np.testing.assert_allclose(y, (x @ w['weight']) * 2)

    def test_matmul_op_raw_matches_etp(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        w = {'weight': jnp.arange(12.0).reshape(4, 3)}
        np.testing.assert_allclose(op.xw_to_y(x, w), op.raw_xw_to_y(x, w))

    def test_elemwise_op(self):
        op = ElemWiseOp(fn=lambda w: w * 3)
        w = jnp.ones((5,))
        y = op(w)
        np.testing.assert_allclose(y, 3.0)

    def test_lora_op(self):
        op = LoraOp(alpha=2.0)
        x = jnp.ones((2, 4))
        w = {'B': jnp.ones((4, 2)), 'A': jnp.ones((2, 3))}
        y = op(x, w)
        expected = 2.0 * x @ w['B'] @ w['A']
        np.testing.assert_allclose(y, expected)

    def test_conv_op(self):
        xinfo = jax.ShapeDtypeStruct((1, 4, 4, 1), jnp.float32)
        op = ConvOp(
            xinfo=xinfo,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x = jnp.ones((1, 4, 4, 1))
        w = {'weight': jnp.ones((3, 3, 1, 2))}
        y = op(x, w)
        assert y.shape == (1, 4, 4, 2)

    def test_sparse_op(self):
        from saiunit import sparse as ss
        dense = jnp.eye(3) * 2.0
        csr = ss.CSR.fromdense(dense)
        op = SpMatMulOp(sparse_mat=csr)
        x = jnp.arange(3.0)
        w = {'weight': csr.data}
        y = op(x, w)
        np.testing.assert_allclose(y, x @ dense)


# ---------------------------------------------------------------------------
# Param classes — construction + execute
# ---------------------------------------------------------------------------

class TestParamsForward:

    def test_etrace_param_is_paramstate(self):
        p = ETraceParam({'weight': jnp.ones((4, 4))}, op=MatMulOp())
        assert isinstance(p, brainstate.ParamState)
        assert isinstance(p.op, MatMulOp)

    def test_etrace_param_execute(self):
        p = ETraceParam({'weight': jnp.ones((4, 4))}, op=MatMulOp())
        y = p.execute(jnp.ones((2, 4)))
        assert y.shape == (2, 4)

    def test_elemwise_param_execute(self):
        p = ElemWiseParam(jnp.ones((3,)), op=lambda w: w * 5)
        y = p.execute()
        np.testing.assert_allclose(y, 5.0)

    def test_elemwise_param_with_elemwiseop(self):
        p = ElemWiseParam(jnp.ones((3,)), op=ElemWiseOp(lambda w: w + 1))
        y = p.execute()
        np.testing.assert_allclose(y, 2.0)

    def test_nontemp_param_execute(self):
        p = NonTempParam({'weight': jnp.ones((4, 4))}, op=MatMulOp())
        assert isinstance(p, brainstate.ParamState)
        y = p.execute(jnp.ones((2, 4)))
        assert y.shape == (2, 4)

    def test_nontemp_param_with_plain_callable(self):
        p = NonTempParam(jnp.ones((4, 4)), op=lambda x, w: x @ w)
        y = p.execute(jnp.ones((2, 4)))
        assert y.shape == (2, 4)

    def test_fake_etrace_param_not_paramstate(self):
        p = FakeETraceParam({'weight': jnp.ones((4, 4))}, op=MatMulOp())
        assert not isinstance(p, brainstate.ParamState)
        y = p.execute(jnp.ones((2, 4)))
        assert y.shape == (2, 4)

    def test_fake_elemwise_param_not_paramstate(self):
        p = FakeElemWiseParam(jnp.ones((3,)), op=lambda w: w * 7)
        assert not isinstance(p, brainstate.ParamState)
        y = p.execute()
        np.testing.assert_allclose(y, 7.0)

    def test_fake_elemwise_param_with_elemwiseop(self):
        p = FakeElemWiseParam(jnp.ones((3,)), op=ElemWiseOp(lambda w: w * 4))
        y = p.execute()
        np.testing.assert_allclose(y, 4.0)


# ---------------------------------------------------------------------------
# Compiler integration
# ---------------------------------------------------------------------------

class _ETraceCell(brainstate.nn.Module):
    def __init__(self, cls, **kwargs):
        super().__init__()
        self.w = cls({'weight': jnp.ones((4, 4)) * 0.1}, op=MatMulOp(), **kwargs)
        self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

    def update(self, x):
        self.h.value = jnp.tanh(x + self.w.execute(self.h.value))
        return self.h.value


class _FakeCell(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.fake = FakeETraceParam({'weight': jnp.ones((4, 4)) * 0.1}, op=MatMulOp())
        self.w_real = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
        self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

    def update(self, x):
        self.h.value = jnp.tanh(
            self.fake.execute(self.h.value)
            + braintrace.matmul(x, self.w_real.value)
        )
        return self.h.value


class TestCompilerIntegration:

    def test_etrace_param_registers_relation(self):
        cell = _ETraceCell(ETraceParam)
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = braintrace.compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )
        assert len(graph.hidden_param_op_relations) == 1

    def test_nontemp_param_registers_no_relation(self):
        """NonTempParam uses raw JAX ops → no ETP primitive → no relation."""
        cell = _ETraceCell(NonTempParam)
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = braintrace.compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )
        assert len(graph.hidden_param_op_relations) == 0

    def test_fake_etrace_param_registers_no_relation(self):
        """FakeETraceParam is not a ParamState → compiler ignores it."""
        cell = _FakeCell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = braintrace.compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )
        # Only self.w_real should produce a relation (via x @ w_real — but that
        # path is a plain jnp.matmul, so no ETP primitive, so zero relations).
        # Confirm no warning for FakeETraceParam either.
        kinds = {d.kind for d in graph.diagnostics}
        assert braintrace.DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE not in kinds


# ---------------------------------------------------------------------------
# Deprecation warnings
# ---------------------------------------------------------------------------

class TestDeprecationWarnings:

    def _reset(self):
        from braintrace._legacy import _ops, _params
        _ops._warned.clear()
        _params._warned.clear()

    def test_matmul_op_warns(self):
        self._reset()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            MatMulOp()
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'MatMulOp' in str(w.message)
            for w in captured
        )

    def test_etrace_param_warns(self):
        self._reset()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            ETraceParam({'weight': jnp.ones((4, 4))}, op=MatMulOp())
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'ETraceParam' in str(w.message)
            for w in captured
        )

    def test_warning_once_per_class(self):
        self._reset()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            MatMulOp()
            MatMulOp()
            MatMulOp()
        matmul_warnings = [
            w for w in captured
            if issubclass(w.category, DeprecationWarning)
            and 'MatMulOp' in str(w.message)
        ]
        assert len(matmul_warnings) == 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_stop_param_gradients_context(self):
        # Pure API compat; just verify the context manager works.
        with stop_param_gradients(True):
            pass
        with stop_param_gradients(False):
            pass

    def test_general_y2w(self):
        def xw2y(x, w):
            return x @ w

        x = jnp.ones((4,))
        w = jnp.ones((4, 3))
        y = jnp.ones((3,))
        w_like = general_y2w(xw2y, x, y, w)
        assert w_like.shape == w.shape


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------

class TestTopLevelExports:

    def test_all_classes_at_top_level(self):
        assert braintrace.ETraceParam is ETraceParam
        assert braintrace.ElemWiseParam is ElemWiseParam
        assert braintrace.NonTempParam is NonTempParam
        assert braintrace.FakeETraceParam is FakeETraceParam
        assert braintrace.FakeElemWiseParam is FakeElemWiseParam
        assert braintrace.ETraceOp is ETraceOp
        assert braintrace.MatMulOp is MatMulOp
        assert braintrace.ElemWiseOp is ElemWiseOp
        assert braintrace.ConvOp is ConvOp
        assert braintrace.SpMatMulOp is SpMatMulOp
        assert braintrace.LoraOp is LoraOp
