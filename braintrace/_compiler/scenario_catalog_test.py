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

"""Discriminative scenario catalog for the ETrace compiler.

Unlike count-based smoke tests, every scenario here asserts:

  1. **Exact relation set** — the ``(weight_path, hidden_path)`` pairs the
     compiler produces, not just ``len(relations)``.
  2. **Type-identity dispatch** — ``r.primitive is <expected primitive>``,
     compared by object identity.
  3. **Structured diagnostics** — when a weight is excluded, the
     corresponding :class:`DiagnosticKind` record must be present.
  4. **Determinism** — compiling the same model twice yields the same
     ``(weight_path, hidden_path)`` ordering.

The scenarios target the three core principles stated in ``CLAUDE.md``:

  - Primitive type identity, not string matching.
  - Only ``etp_elemwise_p``-class primitives are traversable on the tail
    from ``y`` to ``h``.
  - ``W -> non-gradient-enabled W -> h`` excludes the preceding weight.
"""

import warnings

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace import (
    DiagnosticKind,
    DiagnosticLevel,
    compile_etrace_graph,
)
from braintrace._op import (
    etp_conv_p,
    etp_elemwise_p,
    etp_mm_p,
    etp_mv_p,
)


def _compile(model, *inputs):
    """Compile, suppressing expected weight-exclusion UserWarnings.

    Tests still assert on the structured ``DiagnosticKind`` records, so we
    silence the warning-stream duplicate for readability.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return compile_etrace_graph(model, *inputs, include_hidden_perturb=False)


def _relation_set(graph):
    """Return ``{(weight_path, hidden_path)}`` for exact-set assertions."""
    return {
        (r.path, h)
        for r in graph.hidden_param_op_relations
        for h in r.connected_hidden_paths
    }


def _primitive_for(graph, weight_path):
    for r in graph.hidden_param_op_relations:
        if r.path == weight_path:
            return r.primitive
    raise AssertionError(
        f'No relation found for {weight_path!r}; have: '
        f'{[r.path for r in graph.hidden_param_op_relations]}'
    )


def _assert_deterministic(model_factory, inp):
    """Compile twice, check identical ordering of relations and diagnostics.

    ``model_factory`` must return a fully-initialised model; this helper
    calls :func:`brainstate.nn.init_all_states` to make the contract
    explicit and safe against factories that forget.
    """

    def _build():
        m = model_factory()
        brainstate.nn.init_all_states(m)
        return m

    g1 = _compile(_build(), inp)
    g2 = _compile(_build(), inp)
    order1 = [(r.path, tuple(r.connected_hidden_paths))
              for r in g1.hidden_param_op_relations]
    order2 = [(r.path, tuple(r.connected_hidden_paths))
              for r in g2.hidden_param_op_relations]
    assert order1 == order2, (
        f'Relation ordering is not deterministic: {order1} vs {order2}'
    )
    kinds1 = [d.kind for d in g1.diagnostics]
    kinds2 = [d.kind for d in g2.diagnostics]
    assert kinds1 == kinds2, (
        f'Diagnostic record ordering is not deterministic: {kinds1} vs {kinds2}'
    )


# ---------------------------------------------------------------------------
# Category A — Single-primitive baselines
# ---------------------------------------------------------------------------
#
# Each primitive is dispatched by type identity: ``r.primitive is etp_mv_p``,
# not ``r.primitive.name == 'etp_mv_p'``.  These four tests pin down the
# dispatch for the four most common primitives users encounter.

class _UnbatchedMvRNN(brainstate.nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        self.h.value = jnp.tanh(braintrace.matmul(xh, self.w.value))
        return self.h.value


class _BatchedMmRNN(brainstate.nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros((2, n_out)))  # batch size 2

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)  # (batch, n_in+n_out)
        self.h.value = jnp.tanh(braintrace.matmul(xh, self.w.value))
        return self.h.value


class _ElemwiseOnlyRNN(brainstate.nn.Module):
    """``h = tanh(h_prev + element_wise(w))`` — ``etp_elemwise_p``.

    The weight participates only through the identity-like element-wise
    primitive, so the primitive is gradient-enabled and the only
    registered primitive.
    """

    def __init__(self, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        self.h.value = jnp.tanh(self.h.value + x + braintrace.element_wise(self.w.value))
        return self.h.value


class _ConvRNN(brainstate.nn.Module):
    """2-D conv into a hidden state. ``etp_conv_p`` is batched.

    Shapes follow JAX ``conv_general_dilated`` defaults (NCHW / OIHW):
    input ``(N, C_in, H, W)``, kernel ``(C_out, C_in, kH, kW)``.
    """

    def __init__(self):
        super().__init__()
        self.kernel = brainstate.ParamState(
            brainstate.random.randn(2, 1, 3, 3) * 0.1  # C_out, C_in, kH, kW
        )
        self.h = brainstate.HiddenState(jnp.zeros((1, 2, 4, 4)))  # N, C, H, W

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        y = braintrace.conv(x, self.kernel.value, strides=(1, 1))
        self.h.value = jnp.tanh(self.h.value + y)
        return self.h.value


class TestCategoryA_SinglePrimitiveBaselines:

    def test_mv_unbatched(self):
        model = _UnbatchedMvRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        assert _primitive_for(graph, ('w',)) is etp_mv_p
        _assert_deterministic(lambda: _UnbatchedMvRNN(3, 4), inp)

    def test_mm_batched(self):
        model = _BatchedMmRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(2, 3)  # batched

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        assert _primitive_for(graph, ('w',)) is etp_mm_p

    def test_elemwise(self):
        model = _ElemwiseOnlyRNN(4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        assert _primitive_for(graph, ('w',)) is etp_elemwise_p

    def test_conv(self):
        model = _ConvRNN()
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(1, 1, 4, 4)  # N, C_in, H, W

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('kernel',), ('h',))}
        assert _primitive_for(graph, ('kernel',)) is etp_conv_p


# ---------------------------------------------------------------------------
# Category B — Chain traversal (the traversability principle)
# ---------------------------------------------------------------------------
#
# Only ``etp_elemwise_p``-class primitives may sit on the tail from ``y`` to
# ``h``.  A non-gradient-enabled ETP primitive in the middle of the tail is a
# wall: it must sever the relation for the upstream weight.

class _TanhChainRNN(brainstate.nn.Module):
    """``h = tanh(tanh(W@x))`` — only standard JAX ops on the tail."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        y = braintrace.matmul(xh, self.w.value)
        self.h.value = jnp.tanh(jnp.tanh(y))
        return self.h.value


class _ElemwiseChainRNN(brainstate.nn.Module):
    """``h = tanh(element_wise(w2) * matmul(w1, xh))`` — elemwise on the tail
    is allowed (``etp_elemwise_p`` is gradient-enabled), so W1 is included.
    Both W1 and W2 register relations."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w1 = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.w2 = brainstate.ParamState(brainstate.random.randn(n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        y = braintrace.matmul(xh, self.w1.value)
        self.h.value = jnp.tanh(braintrace.element_wise(self.w2.value) * y)
        return self.h.value


class _ElemwiseChainWeightFnRNN(brainstate.nn.Module):
    """``h = tanh(element_wise(w2, weight_fn=tanh) * matmul(w1, xh))`` — the
    gradient-enabled elemwise tail exemption must still include W1 even when
    the elemwise carries a non-identity transform. Both W1 and W2 register."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w1 = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.w2 = brainstate.ParamState(brainstate.random.randn(n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        y = braintrace.matmul(xh, self.w1.value)
        s = braintrace.element_wise(self.w2.value, weight_fn=jnp.tanh)
        self.h.value = jnp.tanh(s * y)
        return self.h.value


class TestElemwiseWeightFnTailExemption:

    def test_weight_fn_elemwise_still_includes_upstream_matmul(self):
        model = _ElemwiseChainWeightFnRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        # Exact relation set: both W1 (upstream matmul) and W2 (transformed
        # elemwise) connect to h.
        assert _relation_set(graph) == {
            (('w1',), ('h',)),
            (('w2',), ('h',)),
        }
        # Primitive identity (NOT name strings) — project principle.
        assert _primitive_for(graph, ('w1',)) is etp_mv_p
        assert _primitive_for(graph, ('w2',)) is etp_elemwise_p


class _TwoMatmulInSeriesRNN(brainstate.nn.Module):
    """``h = tanh(matmul(w2, matmul(w1, xh)))`` — W1's output reaches h only
    via W2's matmul (another non-gradient-enabled ETP primitive). W1 must
    be excluded with a ``RELATION_EXCLUDED_WEIGHT_TO_WEIGHT`` diagnostic.
    W2 is included normally."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w1 = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.w2 = brainstate.ParamState(brainstate.random.randn(n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        mid = braintrace.matmul(xh, self.w1.value)
        self.h.value = jnp.tanh(braintrace.matmul(mid, self.w2.value))
        return self.h.value


class TestCategoryB_ChainTraversal:

    def test_tanh_chain_reaches_weight(self):
        model = _TanhChainRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        assert _primitive_for(graph, ('w',)) is etp_mv_p

    def test_elemwise_chain_includes_both_weights(self):
        model = _ElemwiseChainRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {
            (('w1',), ('h',)),
            (('w2',), ('h',)),
        }
        assert _primitive_for(graph, ('w1',)) is etp_mv_p
        assert _primitive_for(graph, ('w2',)) is etp_elemwise_p

    def test_two_matmul_in_series_excludes_first(self):
        """Principle 3: ``W1 -> W2 -> h`` must drop W1."""
        model = _TwoMatmulInSeriesRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        # Only W2 is registered.
        assert _relation_set(graph) == {(('w2',), ('h',))}
        assert _primitive_for(graph, ('w2',)) is etp_mv_p

        # W1 must emit a structured WEIGHT_TO_WEIGHT exclusion — NOT the
        # weaker NON_TEMPORAL kind.
        w2w = graph.explain(
            weight_path=('w1',),
            kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
        )
        assert len(w2w) == 1, (
            f'W1 must emit exactly one WEIGHT_TO_WEIGHT record; got {w2w}'
        )
        # The blocking primitive reported in context must be the exact
        # object identity of ``etp_mv_p``.
        blocking = w2w[0].context['blocking_primitives']
        assert etp_mv_p in blocking, (
            f'Blocking primitives should include etp_mv_p; got {blocking}'
        )

        # And the NON_TEMPORAL kind must NOT be emitted for W1 — the two
        # reasons have different remediation paths and must stay distinct.
        non_temporal = graph.explain(
            weight_path=('w1',),
            kind=DiagnosticKind.RELATION_EXCLUDED_NON_TEMPORAL,
        )
        assert len(non_temporal) == 0, (
            f'W1 must NOT emit NON_TEMPORAL (it is W->W->h); got {non_temporal}'
        )


# ---------------------------------------------------------------------------
# Category C — Fan-in / fan-out
# ---------------------------------------------------------------------------
#
# The compiler handles (a) two weights summed into a single hidden state
# (both register) and (b) two independent weights driving two independent
# hidden states (each to its own group).

class _TwoWeightsFanInRNN(brainstate.nn.Module):
    """``h = tanh(W1@x + W2@h_prev)`` — both weights flow into the same h."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w_in = brainstate.ParamState(brainstate.random.randn(n_in, n_out))
        self.w_rec = brainstate.ParamState(brainstate.random.randn(n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        y = braintrace.matmul(x, self.w_in.value) + braintrace.matmul(self.h.value, self.w_rec.value)
        self.h.value = jnp.tanh(y)
        return self.h.value


class _IndependentHiddensModel(brainstate.nn.Module):
    """Two disjoint recurrent cells sharing one input stream. Each weight
    reaches exactly one hidden state — the two relations must have
    distinct ``hidden_paths``."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.cell_a = _UnbatchedMvRNN(n_in, n_out)
        self.cell_b = _UnbatchedMvRNN(n_in, n_out)

    def init_state(self, *args, **kwargs):
        self.cell_a.init_state()
        self.cell_b.init_state()

    def update(self, x):
        return self.cell_a.update(x), self.cell_b.update(x)


class TestCategoryC_FanInFanOut:

    def test_two_weights_fan_in(self):
        model = _TwoWeightsFanInRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {
            (('w_in',), ('h',)),
            (('w_rec',), ('h',)),
        }
        for r in graph.hidden_param_op_relations:
            assert r.primitive is etp_mv_p

    def test_independent_hiddens(self):
        model = _IndependentHiddensModel(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {
            (('cell_a', 'w'), ('cell_a', 'h')),
            (('cell_b', 'w'), ('cell_b', 'h')),
        }
        # Each relation must reach exactly one hidden state (not both).
        for r in graph.hidden_param_op_relations:
            assert len(r.connected_hidden_paths) == 1


# ---------------------------------------------------------------------------
# Category D — Exclusion paths
# ---------------------------------------------------------------------------
#
# These scenarios pin down what the compiler must *not* register and why.

class _PlainJaxMatmulRNN(brainstate.nn.Module):
    """``h = tanh(x @ w + h_prev)`` — plain ``@``, no ETP primitive. The
    compiler must register zero relations: ParamState alone is not enough
    without the ETP marker."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        self.h.value = jnp.tanh(x @ self.w.value + self.h.value)
        return self.h.value


class _MixedPlainAndEtpRNN(brainstate.nn.Module):
    """Two weights: one via plain ``@``, one via ``braintrace.matmul``. Only
    the latter is registered — selection is by primitive type, not by
    ParamState presence."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w_plain = brainstate.ParamState(brainstate.random.randn(n_in, n_out))
        self.w_etp = brainstate.ParamState(brainstate.random.randn(n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        plain = x @ self.w_plain.value  # regular JAX — excluded
        etp = braintrace.matmul(self.h.value, self.w_etp.value)  # included
        self.h.value = jnp.tanh(plain + etp)
        return self.h.value


class _NonTemporalWeightRNN(brainstate.nn.Module):
    """``w_loss`` produces a branch that never touches the hidden state
    (only the return value). The compiler must emit
    ``RELATION_EXCLUDED_NON_TEMPORAL`` for it."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w_rec = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.w_loss = brainstate.ParamState(brainstate.random.randn(n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        self.h.value = jnp.tanh(braintrace.matmul(xh, self.w_rec.value))
        # w_loss produces a read-only output; never flows back into h.
        aux = braintrace.matmul(self.h.value, self.w_loss.value)
        return self.h.value, aux


class TestCategoryD_ExclusionPaths:

    def test_plain_jax_matmul_no_relation(self):
        """Principle 1: type-identity dispatch — plain ``@`` must not match."""
        model = _PlainJaxMatmulRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert len(graph.hidden_param_op_relations) == 0, (
            f'Plain @ must not register any ETP relation; got '
            f'{[r.path for r in graph.hidden_param_op_relations]}'
        )

    def test_mixed_only_etp_registered(self):
        model = _MixedPlainAndEtpRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w_etp',), ('h',))}
        # Critically: w_plain must NOT appear in any relation.
        paths = {r.path for r in graph.hidden_param_op_relations}
        assert ('w_plain',) not in paths

    def test_non_temporal_weight_emits_record(self):
        """A weight whose output never reaches a hidden state is
        non-temporal. It must emit a ``RELATION_EXCLUDED_NON_TEMPORAL``
        record (not WEIGHT_TO_WEIGHT — there is no other ETP primitive on
        its path)."""
        model = _NonTemporalWeightRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        # w_rec is included; w_loss is not.
        paths = {r.path for r in graph.hidden_param_op_relations}
        assert ('w_rec',) in paths
        assert ('w_loss',) not in paths

        # Diagnostic: non-temporal, not weight-to-weight.
        non_temporal = graph.explain(
            weight_path=('w_loss',),
            kind=DiagnosticKind.RELATION_EXCLUDED_NON_TEMPORAL,
        )
        assert len(non_temporal) == 1, (
            f'w_loss must emit exactly one NON_TEMPORAL record; got {non_temporal}'
        )
        w2w = graph.explain(
            weight_path=('w_loss',),
            kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
        )
        assert len(w2w) == 0


# ---------------------------------------------------------------------------
# Category E — Canonical recurrences with structured diagnostics
# ---------------------------------------------------------------------------

class TestCategoryE_CanonicalRecurrences:

    def test_gru_wr_structured_exclusion(self):
        """GRU has Wr, Wz, Wh. Only Wz and Wh are included: Wr -> Wh -> h is
        a W->W->h pathway. The exclusion must come with a structured record
        whose ``context.blocking_primitives`` names ``etp_mv_p`` (the Wh
        matmul on 1-D input)."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        graph = _compile(gru, inp)

        # Exact inclusion set — Wr excluded.
        paths = {r.path[0] for r in graph.hidden_param_op_relations}
        assert paths == {'Wz', 'Wh'}, (
            f'GRU must include exactly Wz and Wh; got {paths}'
        )
        for r in graph.hidden_param_op_relations:
            assert r.primitive is etp_mv_p

        # Diagnostic: Wr -> WEIGHT_TO_WEIGHT with etp_mv_p blocking.
        w2w = graph.explain(
            kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
        )
        assert len(w2w) == 1
        rec = w2w[0]
        assert rec.weight_path[0] == 'Wr'
        assert etp_mv_p in rec.context['blocking_primitives']

    def test_lstm_all_four_gates_included(self):
        """LSTM's four gates (Wi, Wf, Wg, Wo) all reach the hidden states —
        the recurrence is a non-parametric composition at the tail. All
        four must register relations."""
        lstm = braintrace.nn.LSTMCell(3, 4)
        brainstate.nn.init_all_states(lstm)
        inp = brainstate.random.rand(3)

        graph = _compile(lstm, inp)

        names = {r.path[0] for r in graph.hidden_param_op_relations}
        # All four gates plus no extras.
        assert 'Wi' in names
        assert 'Wf' in names
        assert 'Wg' in names
        assert 'Wo' in names

        for r in graph.hidden_param_op_relations:
            assert r.primitive is etp_mv_p


# ---------------------------------------------------------------------------
# Category F — Determinism across full graph compilation
# ---------------------------------------------------------------------------

class TestCategoryF_Determinism:

    def test_gru_compile_twice_same_order(self):
        inp = brainstate.random.rand(3)
        _assert_deterministic(lambda: braintrace.nn.GRUCell(3, 4), inp)

    def test_lstm_compile_twice_same_order(self):
        inp = brainstate.random.rand(3)
        _assert_deterministic(lambda: braintrace.nn.LSTMCell(3, 4), inp)

    def test_fan_in_compile_twice_same_order(self):
        inp = brainstate.random.rand(3)
        _assert_deterministic(lambda: _TwoWeightsFanInRNN(3, 4), inp)


# ---------------------------------------------------------------------------
# Category G — Structural smoke test
# ---------------------------------------------------------------------------
#
# After compilation, ``relation.y_to_hidden_groups(y_val, const_vals)`` must
# execute and return tensors whose shape matches the hidden group's
# concatenated-hidden shape.  If this fails, the transition jaxpr is
# malformed and downstream VJP algorithms will crash at runtime.

from braintrace._compiler.diagnostics import diagnostic_context
from braintrace._compiler.hid_param_op import (
    PathClassification,
    _scan_jaxpr_for_etp_eqns,
)
from braintrace._compiler.scenario_catalog import (
    PytreeParamRNN,
    MaskedWeightRNN,
    StackedDeepRNN,
    SharedTiedWeightRNN,
    MixedBatchedRNN,
    PartialPathRNN,
    NestedJitRNN,
    CondBranchRNN,
    ScanBodyRNN,
    make_scan_body_etp_jaxpr,
    make_cond_branches_etp_jaxpr,
    make_while_body_etp_jaxpr,
)


# ---------------------------------------------------------------------------
# Category H — Pytree-valued ParamState
# ---------------------------------------------------------------------------

class TestCategoryH_PytreeWeight:
    """One ``ParamState`` holds ``{'W': ..., 'b': ...}``; only ``W`` is fed
    to the ETP primitive. The compiler must register a single relation
    pointing at the ParamState path and resolve ``weight_leaf_idx``
    to the index of ``W`` in the pytree-leaves enumeration."""

    def test_pytree_weight_resolves_to_paramstate(self):
        model = PytreeParamRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('theta',), ('h',))}
        rel = graph.hidden_param_op_relations[0]
        assert rel.primitive is etp_mv_p
        # The weight tensor we matmul'd has shape (n_in + n_out, n_out).
        assert tuple(rel.trainable_vars['weight'].aval.shape) == (3 + 4, 4)
        # trainable_leaf_indices must point at *some* leaf of the ParamState
        # value (jax.tree.leaves of {'W': ..., 'b': ...} has two leaves).
        assert 0 <= rel.trainable_leaf_indices['weight'] <= 1

    def test_pytree_weight_processing_chain_empty(self):
        """``W`` is consumed directly — no mask/weight_fn equations.
        The ``weight_processing_chain`` must therefore be empty."""
        model = PytreeParamRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)
        rel = graph.hidden_param_op_relations[0]
        assert rel.trainable_processing_chains['weight'] == ()


# ---------------------------------------------------------------------------
# Category I — Masked / processed weight
# ---------------------------------------------------------------------------

class TestCategoryI_MaskedWeight:
    """``mask * w`` flows into the matmul. The compiler must trace the
    weight backward through the elementwise mul and report a non-empty
    ``weight_processing_chain`` so callers can reason about the chain."""

    def test_masked_weight_records_processing_chain(self):
        model = MaskedWeightRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        rel = graph.hidden_param_op_relations[0]
        # The mask multiplication appears in the processing chain.
        chain_names = {p.name for p in rel.trainable_processing_chains['weight']}
        assert 'mul' in chain_names, (
            f'Expected a "mul" primitive in the weight processing chain; '
            f'got {chain_names}'
        )


# ---------------------------------------------------------------------------
# Category J — Stacked deep recurrent network
# ---------------------------------------------------------------------------

class TestCategoryJ_StackedDeep:
    """Three independent recurrent cells. Each weight must reach exactly
    its own cell's hidden state — never the next cell's hidden state.

    The home-group restriction in ``_bfs_forward`` is what makes this
    work; without it, ``cell0``'s weight would also register a relation
    with ``cell1.h`` and ``cell2.h`` (since they are downstream)."""

    def test_each_weight_scoped_to_own_layer(self):
        model = StackedDeepRNN(3, 4, depth=3)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {
            (('cell0', 'w'), ('cell0', 'h')),
            (('cell1', 'w'), ('cell1', 'h')),
            (('cell2', 'w'), ('cell2', 'h')),
        }
        for r in graph.hidden_param_op_relations:
            assert len(r.connected_hidden_paths) == 1
            assert r.primitive is etp_mv_p


# ---------------------------------------------------------------------------
# Category K — Shared / tied weight
# ---------------------------------------------------------------------------

class TestCategoryK_SharedTiedWeight:
    """One ParamState consumed by *two* ``braintrace.matmul`` call sites.
    Two relations expected, both pointing at the same weight_path. The
    relations must remain distinct objects — selection is per-call-site,
    not per-ParamState."""

    def test_two_relations_per_call_site(self):
        model = SharedTiedWeightRNN(4, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)

        graph = _compile(model, inp)

        rels = graph.hidden_param_op_relations
        assert len(rels) == 2, (
            f'Expected exactly two relations for shared weight; got {len(rels)}'
        )
        for r in rels:
            assert r.path == ('w',)
            assert r.connected_hidden_paths == [('h',)]
            assert r.primitive is etp_mv_p
        # The two relations must use different y_var instances (different
        # call sites produce different intermediate vars).
        assert rels[0].y_var is not rels[1].y_var


# ---------------------------------------------------------------------------
# Category L — Mixed batching modes coexisting in one model
# ---------------------------------------------------------------------------

class TestCategoryL_MixedBatching:
    """Compiler dispatches by primitive identity, so a model holding both a
    batched (``etp_mm_p``) and an unbatched (``etp_mv_p``) call must
    register one relation per call with the correct primitive identity."""

    def test_each_relation_uses_its_own_primitive(self):
        model = MixedBatchedRNN(3, 4)
        brainstate.nn.init_all_states(model)
        x_u = brainstate.random.rand(3)
        x_b = brainstate.random.rand(2, 3)

        graph = _compile(model, x_u, x_b)

        by_path = {
            r.path: r.primitive
            for r in graph.hidden_param_op_relations
        }
        assert by_path == {
            ('w_unbatched',): etp_mv_p,
            ('w_batched',): etp_mm_p,
        }


# ---------------------------------------------------------------------------
# Category M — Partial path (MIXED classification)
# ---------------------------------------------------------------------------

class TestCategoryM_PartialPath:
    """``w1`` reaches ``h`` via *both* a direct tail and an indirect path
    that crosses ``w2``. Per the user's requirement we *preserve* the
    historical inclusion behavior for ``w1`` but emit a structured
    ``RELATION_PARTIAL_PATH`` informational record so downstream
    consumers can reason about the partial gradient capture."""

    def test_w1_classified_mixed_w2_classified_direct(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        # Both weights register relations.
        assert _relation_set(graph) == {
            (('w1',), ('h',)),
            (('w2',), ('h',)),
        }

        by_path = {r.path: r for r in graph.hidden_param_op_relations}
        assert by_path[('w1',)].path_classification == {
            ('h',): PathClassification.MIXED,
        }
        assert by_path[('w2',)].path_classification == {
            ('h',): PathClassification.ALL_DIRECT,
        }

    def test_partial_path_diagnostic_emitted(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        partial = graph.explain(
            kind=DiagnosticKind.RELATION_PARTIAL_PATH,
            weight_path=('w1',),
        )
        assert len(partial) == 1, (
            f'w1 must emit exactly one PARTIAL_PATH record; got {partial}'
        )
        assert partial[0].context['classification'] == PathClassification.MIXED

    def test_w2_emits_no_partial_record(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        partial = graph.explain(
            kind=DiagnosticKind.RELATION_PARTIAL_PATH,
            weight_path=('w2',),
        )
        assert len(partial) == 0


# ---------------------------------------------------------------------------
# Category N — Control flow (scan / while / cond)
# ---------------------------------------------------------------------------

class TestCategoryN_ControlFlow:
    """ETP primitives inside ``scan`` / ``while`` / ``cond`` bodies are
    detected by the scanner and reported via
    ``PRIMITIVE_INSIDE_CONTROL_FLOW``. They are *not* lifted into the
    top-level relation set (carry-variable lineage is not yet supported);
    skipping them is the safe behavior, and the structured diagnostic
    surfaces the location for the user.

    These tests exercise the *raw scanner* on unprocessed jaxprs. In the
    full pipeline, ETP-relevant ``cond``/``scan`` equations normally never
    reach the scanner: conds are if-converted into inlined branches +
    ``select_n`` and eligible scans are unrolled at extraction time
    (``_compiler/canonicalize.py``), so their weights DO participate as
    relations — see ``test_cond_branches_full_pipeline_converts`` and
    ``test_scan_body_full_pipeline_unrolls``. Only ineligible scans (too
    long, effectful, while-in-body) and ``while`` still fall through to
    the scanner."""

    def test_cond_branches_full_pipeline_converts(self):
        model = CondBranchRNN(3, 4)
        brainstate.nn.init_all_states(model)
        graph = compile_etrace_graph(model, jnp.ones(3))
        names = [eqn.primitive.name for eqn in graph.module_info.jaxpr.eqns]
        assert 'cond' not in names
        assert 'select_n' in names
        assert len(graph.hidden_param_op_relations) == 2
        kinds = [r.kind for r in graph.diagnostics]
        assert DiagnosticKind.COND_IF_CONVERTED in kinds

    def test_scan_body_full_pipeline_unrolls(self):
        model = ScanBodyRNN(4, loops=3)
        brainstate.nn.init_all_states(model)
        with warnings.catch_warnings():
            # Earlier sub-steps' weights are excluded per the
            # weight->weight->hidden invariant and warn about it.
            warnings.simplefilter('ignore', UserWarning)
            graph = compile_etrace_graph(model, jnp.ones(4))
        names = [eqn.primitive.name for eqn in graph.module_info.jaxpr.eqns]
        assert 'scan' not in names
        # Only the final sub-step's two ETP ops are relations; the earlier
        # sub-steps reach `h` through another trainable ETP op.
        assert len(graph.hidden_param_op_relations) == 2
        kinds = [r.kind for r in graph.diagnostics]
        assert DiagnosticKind.SCAN_UNROLLED in kinds

    def test_scan_body_etp_default_policy_raises(self):
        """Default policy (``etp_in_control_flow='error'``): an ETP primitive
        inside an unflattened scan body is a hard error — its weight would
        otherwise silently drop out of online learning. An ERROR-level
        record is emitted before raising."""
        jaxpr = make_scan_body_etp_jaxpr(3, 4)

        with diagnostic_context() as reporter:
            with pytest.raises(NotImplementedError) as exc_info:
                _scan_jaxpr_for_etp_eqns(jaxpr)

        msg = str(exc_info.value)
        assert 'scan/while/cond' in msg
        assert "etp_in_control_flow='exclude'" in msg
        records = [
            r for r in reporter.records()
            if r.kind is DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW
        ]
        assert len(records) == 1
        assert records[0].level is DiagnosticLevel.ERROR

    def test_scan_body_etp_exclude_policy_warns_and_drops(self):
        """``etp_in_control_flow='exclude'`` pins the pre-Phase-3 behavior:
        a WARNING record and exclusion (no bubbling up), no raise."""
        jaxpr = make_scan_body_etp_jaxpr(3, 4)
        policy = braintrace.ControlFlowPolicy(etp_in_control_flow='exclude')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with diagnostic_context() as reporter:
                top = _scan_jaxpr_for_etp_eqns(jaxpr, policy=policy)

        assert top == [], 'ETP inside scan body must NOT bubble up'
        records = [
            r for r in reporter.records()
            if r.kind is DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW
        ]
        assert len(records) == 1
        assert records[0].level is DiagnosticLevel.WARNING

    def test_cond_branches_etp_exclude_policy_diagnostic_per_branch(self):
        jaxpr = make_cond_branches_etp_jaxpr(3, 4)
        policy = braintrace.ControlFlowPolicy(etp_in_control_flow='exclude')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with diagnostic_context() as reporter:
                top = _scan_jaxpr_for_etp_eqns(jaxpr, policy=policy)

        assert top == []
        n_cf = sum(
            r.kind is DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW
            for r in reporter.records()
        )
        assert n_cf == 2, (
            f'Expected exactly two control-flow records (one per branch); '
            f'got {n_cf}'
        )

    def test_cond_branches_etp_default_policy_raises(self):
        jaxpr = make_cond_branches_etp_jaxpr(3, 4)
        with diagnostic_context():
            with pytest.raises(NotImplementedError):
                _scan_jaxpr_for_etp_eqns(jaxpr)

    def test_while_body_etp_default_policy_raises(self):
        """The weight of an ETP op inside a while body reaches the loop as an
        already-processed value (not a tracked weight invar), so
        ``check_unsupported_op`` cannot catch it — the relation scanner is
        the last line of defence and must raise by default."""
        jaxpr = make_while_body_etp_jaxpr(4, 4)

        with diagnostic_context() as reporter:
            with pytest.raises(NotImplementedError):
                _scan_jaxpr_for_etp_eqns(jaxpr)

        records = [
            r for r in reporter.records()
            if r.kind is DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW
        ]
        assert len(records) == 1
        assert records[0].level is DiagnosticLevel.ERROR

    def test_while_body_etp_exclude_policy_warns_and_drops(self):
        jaxpr = make_while_body_etp_jaxpr(4, 4)
        policy = braintrace.ControlFlowPolicy(etp_in_control_flow='exclude')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with diagnostic_context() as reporter:
                top = _scan_jaxpr_for_etp_eqns(jaxpr, policy=policy)

        assert top == []
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW in kinds

    def test_bogus_etp_in_control_flow_raises_value_error(self):
        jaxpr = make_while_body_etp_jaxpr(4, 4)
        policy = braintrace.ControlFlowPolicy(etp_in_control_flow='bogus')
        with pytest.raises(ValueError, match='etp_in_control_flow'):
            _scan_jaxpr_for_etp_eqns(jaxpr, policy=policy)


# ---------------------------------------------------------------------------
# Category G — Structural smoke test (kept as last category for clarity)
# ---------------------------------------------------------------------------


class TestCategoryG_StructuralSmoke:

    def test_gru_y_to_hidden_groups_executes(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        graph = _compile(gru, inp)

        # ``jaxpr_call`` returns ``(out, etrace_vals, oth_state_vals, temps)``
        # where ``temps`` is the const-var value map for any outvar the
        # compiler added beyond the model's original return.
        _, _, _, temps = graph.module_info.jaxpr_call(inp)

        executed = 0
        for r in graph.hidden_param_op_relations:
            y_val = temps.get(r.y_var)
            if y_val is None:
                continue
            vals = r.y_to_hidden_groups(y_val, temps, concat_hidden_vals=True)
            assert len(vals) == len(r.hidden_groups)
            for v, group in zip(vals, r.hidden_groups):
                # Concatenated hidden has the group's varshape as a prefix.
                assert v.shape[:len(group.varshape)] == tuple(group.varshape)
            executed += 1

        assert executed > 0, (
            'No relation had its y_var available as a temp — '
            'the compiler is not emitting the y values it registered.'
        )


# ---------------------------------------------------------------------------
# Category O — Nested jit boundary
# ---------------------------------------------------------------------------

class TestCategoryO_NestedJit:
    """The ETP matmul lives inside a user ``jax.jit`` function. The
    compiler inlines the jit body at extraction time, so the relation
    must be found exactly as for the plain :class:`UnbatchedMvRNN` —
    same relation set, same primitive identity, executable transition."""

    def test_jit_wrapped_matmul_registers_relation(self):
        model = NestedJitRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        rel = graph.hidden_param_op_relations[0]
        assert rel.primitive is etp_mv_p

    def test_jit_wrapped_matmul_transition_executes(self):
        model = NestedJitRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        _, _, _, temps = graph.module_info.jaxpr_call(inp)
        rel = graph.hidden_param_op_relations[0]
        y_val = temps[rel.y_var]
        vals = rel.y_to_hidden_groups(y_val, temps, concat_hidden_vals=True)
        assert len(vals) == len(rel.hidden_groups)
        for v, group in zip(vals, rel.hidden_groups):
            assert v.shape[:len(group.varshape)] == tuple(group.varshape)

    def test_jit_compile_twice_same_order(self):
        def build():
            model = NestedJitRNN(3, 4)
            brainstate.nn.init_all_states(model)
            return _relation_set(_compile(model, brainstate.random.rand(3)))

        assert build() == build()


# ---------------------------------------------------------------------------
# Category P — Branching fan-out (one y directly feeds two groups)
# ---------------------------------------------------------------------------

from braintrace._compiler.scenario_catalog import (
    BranchingFanOutRNN,
    ConstWeightParamBiasRNN,
)


class TestCategoryP_BranchingFanOut:
    """``y = matmul(x, w)`` directly feeds two *independent* recurrent hidden
    states (two hidden groups). The single relation must record BOTH groups —
    not just the group of whichever hidden outvar the BFS happened to reach
    first."""

    def test_one_relation_records_both_groups(self):
        model = BranchingFanOutRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        rels = graph.hidden_param_op_relations
        assert len(rels) == 1
        r = rels[0]
        assert r.connected_hidden_paths == [('h1',), ('h2',)]
        assert len(r.hidden_groups) == 2
        assert len(r.y_to_hidden_group_jaxprs) == 2
        assert r.primitive is etp_mv_p

    def test_stacked_layers_stay_scoped(self):
        # The directly-fed-groups fix must NOT leak downstream layers back
        # in: cell0's weight still reaches only cell0's hidden state.
        model = StackedDeepRNN(3, 4, depth=2)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {
            (('cell0', 'w'), ('cell0', 'h')),
            (('cell1', 'w'), ('cell1', 'h')),
        }

    def test_fan_out_stable_across_recompiles(self):
        def build():
            model = BranchingFanOutRNN(3, 4)
            brainstate.nn.init_all_states(model)
            g = _compile(model, brainstate.random.rand(3))
            return [
                (r.path, tuple(r.connected_hidden_paths),
                 tuple(grp.index for grp in r.hidden_groups))
                for r in g.hidden_param_op_relations
            ]

        assert build() == build()


# ---------------------------------------------------------------------------
# Category Q — Any-trainable-key gating (const weight, ParamState bias)
# ---------------------------------------------------------------------------

class TestCategoryQ_AnyTrainableKeyGating:
    """The primitive's first trainable key ('weight') is a constant array,
    but the 'bias' key IS a ParamState. The relation must be registered with
    ``bias`` as its only resolved trainable key instead of being vetoed by
    the unresolvable first key."""

    def test_bias_only_relation_registers(self):
        model = ConstWeightParamBiasRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        rels = graph.hidden_param_op_relations
        assert len(rels) == 1
        r = rels[0]
        assert set(r.trainable_paths) == {'bias'}
        assert r.trainable_paths['bias'] == ('b',)
        assert r.connected_hidden_paths == [('h',)]

    def test_no_paramstate_at_all_still_excluded(self):
        # Both weight and bias constant: the exclusion diagnostic must
        # be preserved.
        class _AllConstRNN(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w_const = jnp.asarray(brainstate.random.randn(7, 4))
                self.h = brainstate.HiddenState(jnp.zeros(4))

            def init_state(self, *args, **kwargs):
                self.h.value = jnp.zeros_like(self.h.value)

            def update(self, x):
                xh = jnp.concatenate([x, self.h.value])
                self.h.value = jnp.tanh(braintrace.matmul(xh, self.w_const))
                return self.h.value

        model = _AllConstRNN()
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        assert len(graph.hidden_param_op_relations) == 0
        assert len(graph.explain(kind=DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE)) == 1


# ---------------------------------------------------------------------------
# Category R — Complex module graphs (shared submodule, residual, deep nesting)
# ---------------------------------------------------------------------------

from braintrace._compiler.scenario_catalog import (
    SharedSubmoduleTwiceRNN,
    ResidualSkipRNN,
    DeepNestedModuleRNN,
)


class TestCategoryR_ComplexModuleGraphs:

    def test_shared_submodule_two_call_sites(self):
        # One Linear module (one ParamState) applied at two call sites per
        # step: two distinct relations, both pointing at the shared weight
        # path — the module-sharing analogue of the tied-weight scenario.
        model = SharedSubmoduleTwiceRNN(4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)

        graph = _compile(model, inp)

        rels = graph.hidden_param_op_relations
        assert len(rels) == 2
        for r in rels:
            assert r.path == ('proj', 'weight')
            assert r.connected_hidden_paths == [('h',)]
        assert rels[0].y_var is not rels[1].y_var

    def test_residual_skip_does_not_change_relation(self):
        # The elementwise residual add sits on the non-parametric tail: the
        # relation for ``w`` is unchanged and the skip path creates nothing.
        model = ResidualSkipRNN(4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)

        graph = _compile(model, inp)

        assert _relation_set(graph) == {(('w',), ('h',))}
        rel = graph.hidden_param_op_relations[0]
        assert rel.primitive is etp_mv_p
        assert rel.path_classification == {('h',): PathClassification.ALL_DIRECT}

    def test_residual_transition_includes_skip_add(self):
        # dh/dy must run through the residual add: perturbing y shifts h by
        # tanh'(y + x) — structurally, the transition jaxpr executes and
        # produces the group-shaped output.
        model = ResidualSkipRNN(4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(4)

        graph = _compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)
        rel = graph.hidden_param_op_relations[0]
        vals = rel.y_to_hidden_groups(temps[rel.y_var], temps, concat_hidden_vals=True)
        assert len(vals) == 1
        assert vals[0].shape[:len(rel.hidden_groups[0].varshape)] == tuple(
            rel.hidden_groups[0].varshape
        )

    def test_deep_nested_module_paths(self):
        # Four levels of module nesting: relation and hidden paths carry the
        # full nested module path.
        model = DeepNestedModuleRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = brainstate.random.rand(3)

        graph = _compile(model, inp)

        deep = ('l1', 'inner', 'inner', 'inner')
        assert _relation_set(graph) == {(deep + ('w',), deep + ('h',))}
        rel = graph.hidden_param_op_relations[0]
        assert rel.primitive is etp_mv_p
        assert len(rel.hidden_groups) == 1
        assert rel.hidden_groups[0].hidden_paths == [deep + ('h',)]

    def test_complex_graphs_deterministic(self):
        for factory, n_inp in [
            (lambda: SharedSubmoduleTwiceRNN(4), 4),
            (lambda: ResidualSkipRNN(4), 4),
            (lambda: DeepNestedModuleRNN(3, 4), 3),
        ]:
            def build():
                m = factory()
                brainstate.nn.init_all_states(m)
                g = _compile(m, brainstate.random.rand(n_inp))
                return [
                    (r.path, tuple(r.connected_hidden_paths))
                    for r in g.hidden_param_op_relations
                ]

            assert build() == build()


# ---------------------------------------------------------------------------
# Category S — While-hidden opaque forward (Phase 3)
# ---------------------------------------------------------------------------

from braintrace._compiler.scenario_catalog import (
    WhileSettleRNN,
    WhileSettleTwinRNN,
)


class TestCategoryS_WhileHiddenOpaqueFwd:
    """A **weight-free** ``lax.while_loop`` reads and updates the hidden
    state (``WhileSettleRNN``). Under the default policy
    (``while_hidden='opaque-fwd'``) the compiler keeps the loop as an opaque
    forward node: the ``win`` relation is registered with the while on its
    ``y``-to-hidden tail, hidden-to-hidden Jacobians switch to forward mode,
    and the perturbation pass detaches the loop's inputs with
    ``stop_gradient`` so the perturbed jaxpr stays reverse-traceable.

    Learning-signal correctness note (recorded from ``vjp_base.py``): the
    single-step VJP path consumes ONLY the perturbation cotangents
    (``dg_hid_perturb_or_dl2h``) as the learning signal; the residual
    hidden-input cotangent (``dg_last_hiddens``) is consumed only by the
    multi-step branch. Detaching the while's inputs in the perturbed jaxpr
    therefore does not alter the single-step learning signal — ``dL/dε``
    stays exact because the ``h = fresh + ε`` add sits outside the detach.
    """

    def _build(self, **compile_kwargs):
        model = WhileSettleRNN(3, 4)
        brainstate.nn.init_all_states(model)
        x = brainstate.random.rand(3)
        graph = compile_etrace_graph(model, x, **compile_kwargs)
        return model, x, graph

    def test_compiles_with_single_relation_through_while(self):
        model, x, graph = self._build()
        assert _relation_set(graph) == {(('win',), ('h',))}
        rel = graph.hidden_param_op_relations[0]
        assert rel.primitive is etp_mv_p
        prim_names = {
            eqn.primitive.name
            for jaxpr in rel.y_to_hidden_group_jaxprs
            for eqn in jaxpr.eqns
        }
        assert 'while' in prim_names

    def test_opaque_fwd_diagnostic_emitted(self):
        _, _, graph = self._build()
        kinds = [r.kind for r in graph.diagnostics]
        assert DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD in kinds

    def test_perturbed_jaxpr_detaches_while_inputs(self):
        _, _, graph = self._build()
        assert graph.hidden_perturb is not None
        names = [
            eqn.primitive.name
            for eqn in graph.hidden_perturb.perturb_jaxpr.jaxpr.eqns
        ]
        assert 'while' in names
        assert 'stop_gradient' in names

    def test_error_policy_escape_hatch(self):
        model = WhileSettleRNN(3, 4)
        brainstate.nn.init_all_states(model)
        x = brainstate.random.rand(3)
        with pytest.raises(NotImplementedError):
            compile_etrace_graph(
                model, x,
                control_flow=braintrace.ControlFlowPolicy(while_hidden='error'),
            )

    def test_compiled_values_match_hand_composed_twin(self):
        model, x, graph = self._build()
        twin = WhileSettleTwinRNN(3, 4)
        brainstate.nn.init_all_states(twin)
        twin.win.value = model.win.value

        out, _, _, _ = graph.module_info.jaxpr_call(x)
        expected = twin.update(x)
        out_leaves = jax.tree.leaves(out)
        assert len(out_leaves) == 1
        assert jnp.allclose(out_leaves[0], expected, atol=1e-6)

    def test_compile_is_deterministic(self):
        def build():
            _, _, g = self._build()
            return [
                (r.path, tuple(r.connected_hidden_paths))
                for r in g.hidden_param_op_relations
            ]

        assert build() == build()
