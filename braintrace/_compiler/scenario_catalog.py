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

"""Reusable scenario fixtures for ETrace compiler tests.

Every fixture is a small, fully-defined ``brainstate.nn.Module`` that
exercises a specific, named compiler concern. They are kept here (not in
test files) so they can be imported by the discriminative scenario tests,
the property tests, and the numerical-oracle tests.

Scenario taxonomy
-----------------

The scenarios target the structural axes the compiler must handle
correctly. Each axis has at least one positive scenario (compiler should
include the relation) and at least one negative scenario (compiler should
exclude with a specific structured diagnostic).

* Single-primitive baselines (``UnbatchedMvRNN``, ``BatchedMmRNN``,
  ``ElemwiseOnlyRNN``, ``ConvRNN``).
* Chain traversal (``TanhChainRNN``, ``ElemwiseChainRNN``,
  ``TwoMatmulInSeriesRNN``).
* Fan-in / fan-out (``TwoWeightsFanInRNN``, ``IndependentHiddensModel``).
* Exclusion (``PlainJaxMatmulRNN``, ``MixedPlainAndEtpRNN``,
  ``NonTemporalWeightRNN``).
* Pytree weight (``PytreeParamRNN``).
* Masked weight (``MaskedWeightRNN``).
* Stacked deep (``StackedDeepRNN``).
* Shared / tied weight (``SharedTiedWeightRNN``).
* Mixed batching (``MixedBatchedRNN``).
* Partial path (``PartialPathRNN``).
* Control flow (``ScanBodyEtpRNN``, ``CondBranchEtpRNN``,
  ``WhileBodyEtpRNN``).
* Nested jit (``NestedJitRNN``).
* Branching fan-out (``BranchingFanOutRNN``).
* Shared submodule, two call sites (``SharedSubmoduleTwiceRNN``).
* Residual connection (``ResidualSkipRNN``).
* Deep module nesting (``DeepNestedModuleRNN``).
"""



import brainstate
import jax
import jax.numpy as jnp

import braintrace


# ---------------------------------------------------------------------------
# Single-primitive baselines
# ---------------------------------------------------------------------------

class UnbatchedMvRNN(brainstate.nn.Module):
    """``h = tanh(matmul(concat(x, h_prev), W))`` -> ``etp_mv_p``."""

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


class BatchedMmRNN(brainstate.nn.Module):
    """Batched variant of :class:`UnbatchedMvRNN` -> ``etp_mm_p``."""

    def __init__(self, n_in: int, n_out: int, batch: int = 2):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros((batch, n_out)))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        self.h.value = jnp.tanh(braintrace.matmul(xh, self.w.value))
        return self.h.value


class ElemwiseOnlyRNN(brainstate.nn.Module):
    """``h = tanh(h_prev + x + element_wise(w))`` -> ``etp_elemwise_p``."""

    def __init__(self, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        self.h.value = jnp.tanh(
            self.h.value + x + braintrace.element_wise(self.w.value)
        )
        return self.h.value


# ---------------------------------------------------------------------------
# Pytree weights (ParamState wrapping a dict)
# ---------------------------------------------------------------------------

class PytreeParamRNN(brainstate.nn.Module):
    """ParamState holds a dict ``{'W': ..., 'b': ...}``; only ``W`` is used
    via ETP. The compiler must resolve the right pytree leaf and report
    ``weight_leaf_idx`` consistent with ``jax.tree.leaves`` ordering.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.theta = brainstate.ParamState({
            'W': brainstate.random.randn(n_in + n_out, n_out),
            'b': brainstate.random.randn(n_out),
        })
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        y = braintrace.matmul(xh, self.theta.value['W'])
        self.h.value = jnp.tanh(y + self.theta.value['b'])
        return self.h.value


# ---------------------------------------------------------------------------
# Masked weight (weight passes through a non-trivial processing chain)
# ---------------------------------------------------------------------------

class MaskedWeightRNN(brainstate.nn.Module):
    """``h = tanh(matmul(xh, mask * w))`` — mask is a constant array. The
    compiler must trace through the elementwise multiplication and report
    a non-empty ``weight_processing_chain``.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        # Sparse-style mask: zeros out half the weights structurally.
        self.mask = jnp.where(
            brainstate.random.rand(n_in + n_out, n_out) > 0.5, 1.0, 0.0,
        )
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        masked = self.mask * self.w.value
        self.h.value = jnp.tanh(braintrace.matmul(xh, masked))
        return self.h.value


# ---------------------------------------------------------------------------
# Stacked deep RNN (multiple groups, each weight scoped to its own layer)
# ---------------------------------------------------------------------------

class StackedDeepRNN(brainstate.nn.Module):
    """Three independent recurrent cells stacked. Each weight must reach
    only the hidden state of the cell that owns it — never the next
    layer's hidden state.
    """

    def __init__(self, n_in: int, n_out: int, depth: int = 3):
        super().__init__()
        self.cells = [
            UnbatchedMvRNN(n_in if i == 0 else n_out, n_out)
            for i in range(depth)
        ]
        # Promote child cells to attributes so brainstate finds them.
        for i, c in enumerate(self.cells):
            setattr(self, f'cell{i}', c)

    def init_state(self, *args, **kwargs):
        for c in self.cells:
            c.init_state()

    def update(self, x):
        out = x
        for c in self.cells:
            out = c.update(out)
        return out


# ---------------------------------------------------------------------------
# Shared / tied weight (one ParamState used by two ETP primitives)
# ---------------------------------------------------------------------------

class SharedTiedWeightRNN(brainstate.nn.Module):
    """One ParamState consumed by two ``braintrace.matmul`` calls. The
    compiler must register one relation per call site (two relations
    total), both pointing at the same weight_path.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        # Square so the weight can be applied to both x (size n_in=n_out)
        # and h (size n_out).
        assert n_in == n_out, 'SharedTiedWeightRNN requires n_in == n_out'
        self.w = brainstate.ParamState(brainstate.random.randn(n_in, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        a = braintrace.matmul(x, self.w.value)
        b = braintrace.matmul(self.h.value, self.w.value)
        self.h.value = jnp.tanh(a + b)
        return self.h.value


# ---------------------------------------------------------------------------
# Mixed batching (one ETP op per batching mode in the same model)
# ---------------------------------------------------------------------------

class MixedBatchedRNN(brainstate.nn.Module):
    """Two recurrent paths share an input but use different batching
    modes: one ``etp_mv_p`` (unbatched) and one ``etp_mm_p`` via vmap.
    Compiler must dispatch each by primitive identity.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w_unbatched = brainstate.ParamState(
            brainstate.random.randn(n_in + n_out, n_out)
        )
        self.w_batched = brainstate.ParamState(
            brainstate.random.randn(n_in + n_out, n_out)
        )
        self.h_u = brainstate.HiddenState(jnp.zeros(n_out))
        self.h_b = brainstate.HiddenState(jnp.zeros((2, n_out)))

    def init_state(self, *args, **kwargs):
        self.h_u.value = jnp.zeros_like(self.h_u.value)
        self.h_b.value = jnp.zeros_like(self.h_b.value)

    def update(self, x_unbatched, x_batched):
        xu = jnp.concatenate([x_unbatched, self.h_u.value])
        self.h_u.value = jnp.tanh(braintrace.matmul(xu, self.w_unbatched.value))
        xb = jnp.concatenate([x_batched, self.h_b.value], axis=-1)
        self.h_b.value = jnp.tanh(braintrace.matmul(xb, self.w_batched.value))
        return self.h_u.value, self.h_b.value


# ---------------------------------------------------------------------------
# Partial path (MIXED classification — both direct and via-other-ETP routes)
# ---------------------------------------------------------------------------

class PartialPathRNN(brainstate.nn.Module):
    r"""``mid = matmul(xh, w1); h = tanh(mid + matmul(mid, w2))``.

    ``w1`` reaches ``h`` along *two* paths:
      1. directly through the addition (no other ETP),
      2. indirectly through ``w2``'s matmul.

    Path classification must report ``MIXED`` for ``w1``. ``w2`` is
    classified ``ALL_DIRECT``.
    """

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
        through = braintrace.matmul(mid, self.w2.value)
        self.h.value = jnp.tanh(mid + through)
        return self.h.value


# ---------------------------------------------------------------------------
# Control-flow scenarios — ETP inside scan/while/cond bodies
# ---------------------------------------------------------------------------

def make_scan_body_etp_jaxpr(n_in: int, n_out: int, length: int = 4):
    """Return a jaxpr containing ``braintrace.matmul`` inside ``lax.scan``.

    The compiler's full pipeline rejects ETP inside scan at the
    ``hidden_group`` stage, so this scenario is exercised by feeding the
    raw jaxpr to the lower-level ``_scan_jaxpr_for_etp_eqns``.
    """
    w = jnp.zeros((n_in, n_out))
    xs = jnp.zeros((length, n_in))

    def f(w, xs):
        def body(carry, x_t):
            new_carry = carry + braintrace.matmul(x_t, w)
            return new_carry, new_carry

        carry, _ = jax.lax.scan(body, jnp.zeros((n_out,)), xs)
        return jnp.tanh(carry)

    return jax.make_jaxpr(f)(w, xs).jaxpr


def make_cond_branches_etp_jaxpr(n_in: int, n_out: int):
    """Return a jaxpr containing ``braintrace.matmul`` in *both* cond branches."""
    w_true = jnp.zeros((n_in, n_out))
    w_false = jnp.zeros((n_in, n_out))
    x = jnp.zeros(n_in)

    def f(pred, x, wt, wf):
        return jax.lax.cond(
            pred,
            lambda v: braintrace.matmul(v, wt),
            lambda v: braintrace.matmul(v, wf),
            x,
        )

    return jax.make_jaxpr(f)(True, x, w_true, w_false).jaxpr


def make_while_body_etp_jaxpr(n_in: int, n_out: int, n_iter: int = 3):
    """Return a jaxpr containing ``braintrace.matmul`` inside the body of
    ``lax.while_loop``."""
    assert n_in == n_out, 'while-body fixture requires square w'
    w = jnp.zeros((n_in, n_out))
    x = jnp.zeros(n_in)

    def f(w, x):
        def cond_fn(state):
            i, _ = state
            return i < n_iter

        def body_fn(state):
            i, h = state
            return i + 1, h + braintrace.matmul(x, w)

        _, h_final = jax.lax.while_loop(cond_fn, body_fn, (0, jnp.zeros(n_out)))
        return jnp.tanh(h_final)

    return jax.make_jaxpr(f)(w, x).jaxpr


# ---------------------------------------------------------------------------
# Nested jit — ETP primitive inside a user ``jax.jit`` boundary
# ---------------------------------------------------------------------------

class NestedJitRNN(brainstate.nn.Module):
    """The ETP matmul is wrapped in a user ``jax.jit`` function. The compiler
    inlines the jit body at extraction time, so the relation must be found
    exactly as in :class:`UnbatchedMvRNN`.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in + n_out, n_out))
        self.h = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        @jax.jit
        def proj(xh, w):
            return braintrace.matmul(xh, w)

        xh = jnp.concatenate([x, self.h.value])
        self.h.value = jnp.tanh(proj(xh, self.w.value))
        return self.h.value


# ---------------------------------------------------------------------------
# Branching fan-out — one ETP output directly feeds two uncoupled hiddens
# ---------------------------------------------------------------------------

class BranchingFanOutRNN(brainstate.nn.Module):
    """``y = matmul(x, w)`` feeds two *independent* recurrent hidden states.

    ``h1`` and ``h2`` never read each other, so they land in two different
    hidden groups; both are fed *directly* by ``y`` (no other hidden outvar
    on the path), so the single relation must record **both** groups.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(n_in, n_out))
        self.h1 = brainstate.HiddenState(jnp.zeros(n_out))
        self.h2 = brainstate.HiddenState(jnp.zeros(n_out))

    def init_state(self, *args, **kwargs):
        self.h1.value = jnp.zeros_like(self.h1.value)
        self.h2.value = jnp.zeros_like(self.h2.value)

    def update(self, x):
        y = braintrace.matmul(x, self.w.value)
        self.h1.value = jnp.tanh(0.9 * self.h1.value + y)
        self.h2.value = jnp.tanh(0.5 * self.h2.value - y)
        return self.h1.value + self.h2.value


# ---------------------------------------------------------------------------
# Shared submodule — one module instance called at two sites per step
# ---------------------------------------------------------------------------

class SharedSubmoduleTwiceRNN(brainstate.nn.Module):
    """One ``UnbatchedMvRNN``-style projection module applied twice per step.

    The single ``ParamState`` appears in two ETP equations (two call sites),
    so the compiler must register two relations sharing one weight path —
    the module-sharing analogue of :class:`SharedTiedWeightRNN`.
    """

    def __init__(self, n: int):
        super().__init__()
        self.proj = braintrace.nn.Linear(n, n, b_init=None)
        self.h = brainstate.HiddenState(jnp.zeros(n))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        a = self.proj(x)
        b = self.proj(self.h.value)
        self.h.value = jnp.tanh(a + b)
        return self.h.value


# ---------------------------------------------------------------------------
# Residual connection — skip path around the recurrent projection
# ---------------------------------------------------------------------------

class ResidualSkipRNN(brainstate.nn.Module):
    """``h = tanh(matmul(xh, w) + x_proj_skip)`` with an elementwise residual.

    The residual add is on the non-parametric tail, so the relation for
    ``w`` is unaffected by the skip path; the skip input reaches the hidden
    state without any ETP op and must not create extra relations.
    """

    def __init__(self, n: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(2 * n, n))
        self.h = brainstate.HiddenState(jnp.zeros(n))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value])
        self.h.value = jnp.tanh(braintrace.matmul(xh, self.w.value) + x)
        return self.h.value


# ---------------------------------------------------------------------------
# Deep module nesting — the recurrent cell sits four levels deep
# ---------------------------------------------------------------------------

class _NestingLevel(brainstate.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def init_state(self, *args, **kwargs):
        self.inner.init_state()

    def update(self, x):
        return self.inner.update(x)


class DeepNestedModuleRNN(brainstate.nn.Module):
    """An :class:`UnbatchedMvRNN` wrapped in three pass-through modules.

    Relation and group paths must reflect the full nested module path.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.l1 = _NestingLevel(_NestingLevel(_NestingLevel(UnbatchedMvRNN(n_in, n_out))))

    def init_state(self, *args, **kwargs):
        self.l1.init_state()

    def update(self, x):
        return self.l1.update(x)
