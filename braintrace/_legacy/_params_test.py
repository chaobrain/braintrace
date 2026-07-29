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

"""Smoke tests for the v0.1.x param compatibility shims.

Coverage:
* every legacy param constructs and executes forward.
* ``ETraceParam`` + ``MatMulOp`` produces *one* compiler relation.
* ``NonTempParam`` + ``MatMulOp`` produces *zero* compiler relations.
* ``FakeETraceParam`` is invisible to the compiler.
* ``DeprecationWarning`` fires on package-root param access.
* every legacy param is re-exported at the package root.
"""



import warnings

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace
from braintrace._legacy import (
    ElemWiseOp,
    ElemWiseParam,
    ETraceParam,
    FakeElemWiseParam,
    FakeETraceParam,
    MatMulOp,
    NonTempParam,
)


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
    # As of 0.2.0 the deprecation warning fires at package-root attribute access
    # (``braintrace.ETraceParam``) via ``braintrace.__getattr__``, not at construction
    # and not when importing from the private ``braintrace._legacy`` submodule.

    def test_etrace_param_access_warns(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            _ = braintrace.ETraceParam
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'ETraceParam' in str(w.message)
            for w in captured
        )


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------

class TestTopLevelExports:

    def test_param_classes_at_top_level(self):
        assert braintrace.ETraceParam is ETraceParam
        assert braintrace.ElemWiseParam is ElemWiseParam
        assert braintrace.NonTempParam is NonTempParam
        assert braintrace.FakeETraceParam is FakeETraceParam
        assert braintrace.FakeElemWiseParam is FakeElemWiseParam
