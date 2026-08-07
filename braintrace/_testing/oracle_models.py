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

"""Deterministic toy models for the gradient oracle (test support)."""

from dataclasses import dataclass
from typing import Callable, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace


@dataclass(frozen=True)
class ModelSpec:
    """A zero-arg model factory plus metadata about its parameters.

    ``factory()`` returns a freshly constructed, *uninitialized* model with
    deterministic weights. Callers must call
    ``brainstate.nn.init_all_states(model, batch_size=...)`` themselves.

    Attributes
    ----------
    factory : Callable[[], brainstate.nn.Module]
        Deterministic zero-arg model constructor.
    etp_param_keys : tuple of tuple
        Parameter paths routed through an ETP primitive.
    plain_param_keys : tuple of tuple
        Parameter paths used via plain JAX ops, hence excluded from ETP.
    input_scale : float, optional
        Multiplier applied by :meth:`make_inputs`. Spiking models need a scale
        well above 1.0 to reach threshold at all; below it the loss and the
        gradient are identically zero and any comparison is vacuous (F-25).
    batched_input : bool, optional
        Whether :meth:`make_inputs` emits a leading batch axis of 1. SNN layers
        concatenate the input with the recurrent spike vector, so their ranks
        must match; the rate models broadcast instead and do not need it.
    """

    factory: Callable[[], brainstate.nn.Module]
    etp_param_keys: Tuple[tuple, ...]    # routed through an ETP primitive
    plain_param_keys: Tuple[tuple, ...]  # used via plain JAX ops (excluded from ETP)
    input_scale: float = 1.0
    batched_input: bool = False

    def make_inputs(self, T: int, n_in: int, *, seed: int = 0):
        """Build a ``(T, [1,] n_in)`` input sequence at this spec's scale.

        Values are non-negative so that spiking models receive net excitatory
        drive; a zero-mean drive largely cancels and leaves the network silent.

        Parameters
        ----------
        T : int
            Number of time steps.
        n_in : int
            Input dimension.
        seed : int, optional
            Seed for the input draw.

        Returns
        -------
        jax.Array
            The input sequence, scaled by :attr:`input_scale`.
        """
        rng = np.random.RandomState(seed)
        shape = (T, 1, n_in) if self.batched_input else (T, n_in)
        return self.input_scale * jnp.asarray(np.abs(rng.randn(*shape)).astype('float32'))


def tanh_rnn(n_in: int = 3, n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Batched (batch=1) tanh RNN: recurrent ETP weight ``w``, plain input weight ``win``."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_rec, n_rec))
                )
                self.win = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                inp = x @ self.win.value  # plain op -> excluded from ETP
                self.h.value = jax.nn.tanh(
                    inp + braintrace.matmul(self.h.value, self.w.value)
                )
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def leaky_linear(n_in: int = 3, n_rec: int = 4, leak: float = 0.9, seed: int = 0) -> ModelSpec:
    """Pure leaky integrator with a trainable ETP *input* weight.

    The recurrence ``h_t = leak * h_{t-1} + matmul(x_t, w)`` has hidden-to-hidden
    Jacobian ``leak * I`` exactly (no off-diagonal recurrent term). This is the
    degenerate regime in which rules that discard ``hid2hid_jac`` and assume a
    scalar leak become exact, which makes it the reference model for the
    ``scalar_leak`` temporal recursion. ``w`` reaches every future hidden state
    through the leaky carry, so it is a genuine ETP relation despite being an
    input projection.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                drive = braintrace.matmul(x.reshape(1, -1), self.w.value)
                self.h.value = leak * self.h.value + drive
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def stacked_tanh_rnn(n_in: int = 3, n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Two-layer tanh RNN with two trainable ETP recurrent weights.

    Layer 1: ``h1 = tanh(x @ win + matmul(h1, w1))``; layer 2:
    ``h2 = tanh(h1 @ wmid + matmul(h2, w2))``. ``w1``/``w2`` are ETP recurrent
    weights (two HiddenParamOp relations); ``win``/``wmid`` are plain projections
    (excluded from ETP). Exercises multi-relation D_RTRL == BPTT.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                k = jax.random.PRNGKey
                self.w1 = brainstate.ParamState(0.1 * jax.random.normal(k(seed), (n_rec, n_rec)))
                self.w2 = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 1), (n_rec, n_rec)))
                self.win = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 2), (n_in, n_rec)))
                self.wmid = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 3), (n_rec, n_rec)))
                self.h1 = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.h2 = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                self.h1.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h1.value, self.w1.value)
                )
                self.h2.value = jax.nn.tanh(
                    self.h1.value @ self.wmid.value + braintrace.matmul(self.h2.value, self.w2.value)
                )
                return self.h2.value

        return Net()

    return ModelSpec(
        factory=factory,
        etp_param_keys=(('w1',), ('w2',)),
        plain_param_keys=(('win',), ('wmid',)),
    )


def two_state_rnn(n_in: int = 3, n_rec: int = 3, seed: int = 0) -> ModelSpec:
    """Two coupled hidden states (v, a) that the compiler groups into ONE
    HiddenGroup with ``num_state == 2`` (an LIF+adaptation-like topology).

    ``v_t = 0.9 v + matmul(x, w) - 0.1 a``; ``a_t = 0.95 a + v``. ``w`` is the
    single trainable ETP input weight. D_RTRL handles this exactly; any rule
    whose per-step formulation assumes a single-state group cannot represent it.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_in, n_rec))
                )
                self.v = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.a = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                v, a = self.v.value, self.a.value
                self.v.value = 0.9 * v + braintrace.matmul(x.reshape(1, -1), self.w.value) - 0.1 * a
                self.a.value = 0.95 * a + v
                return self.v.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def batched_tanh_rnn(n_in: int = 3, n_rec: int = 4, batch: int = 4, seed: int = 0) -> ModelSpec:
    """A tanh RNN whose hidden state carries an explicit leading batch axis of
    size ``batch``. The existing models hardcode a size-1 batch, so this one is
    used to exercise batch-axis invariance (batched gradient == sum of
    per-sequence gradients). ``w`` is the recurrent ETP weight; ``win`` is a
    plain input projection.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.5 * jax.random.normal(jax.random.PRNGKey(seed), (n_rec, n_rec))
                )
                self.win = brainstate.ParamState(
                    0.5 * jax.random.normal(jax.random.PRNGKey(seed + 1), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((batch, n_rec)))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h.value, self.w.value)
                )
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def tied_weight_rnn(n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Tanh RNN whose single weight is consumed by TWO ETP matmuls.

    ``h = tanh(matmul(x, w) + matmul(h, w))`` — one ParamState, two call
    sites, so the compiler registers two relations sharing one weight path.
    Locks the multi-eqn-per-weight invariant the scan-unrolling pass depends
    on: trace state is keyed per relation instance (``id(y_var)``, group) and
    per-path gradient contributions accumulate across relations. Exact
    algorithms must match BPTT element-wise (verified bit-exact at adoption).
    Requires ``x`` with ``n_rec`` features (square weight applied to both).
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_rec, n_rec)
                    )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                a = braintrace.matmul(x.reshape(1, -1), self.w.value)
                b = braintrace.matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(a + b)
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def cond_gate_rnn(n_in: int = 3, n_rec: int = 4, leak: float = 0.9, seed: int = 0) -> ModelSpec:
    """Leaky integrator whose drive is a ``lax.cond`` between two ETP matmuls.

    The ETP primitives live inside the ``cond`` branches; the compiler
    if-converts the equation into both inlined branches + ``select_n`` at
    extraction time (Phase 1 canonicalization), so ``w_a`` and ``w_b`` are
    both genuine ETP relations. The hidden-to-hidden Jacobian stays
    ``leak * I`` (the drive contains no ``h``), keeping D_RTRL exact.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w_a = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_in, n_rec)
                    )
                    self.w_b = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_in, n_rec)
                    )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                x_row = x.reshape(1, -1)
                drive = jax.lax.cond(
                    jnp.sum(x) > 0.,
                    lambda: braintrace.matmul(x_row, self.w_a.value),
                    lambda: braintrace.matmul(x_row, self.w_b.value),
                )
                self.h.value = leak * self.h.value + jnp.tanh(drive)
                return self.h.value

        return Net()

    return ModelSpec(
        factory=factory, etp_param_keys=(('w_a',), ('w_b',)), plain_param_keys=()
    )


def scan_body_rnn(n_rec: int = 4, loops: int = 3, seed: int = 0) -> ModelSpec:
    """Tanh RNN whose per-step update is an inner ``for_loop`` of sub-steps.

    Each of the ``loops`` sub-steps applies two ETP matmuls
    (``h <- tanh(matmul(x, w) + matmul(h, w))``), all inside a
    ``brainstate.transform.for_loop`` that lowers to ``lax.scan``. The
    compiler unrolls the inner scan at extraction time (Phase 2
    canonicalization), after which only the *last* sub-step's ETP ops become
    relations — earlier sub-steps reach the hidden state through another
    trainable ETP op (the weight->weight->hidden invariant). Exact algorithms
    must match BPTT element-wise on the unrolled program.
    Requires ``x`` with ``n_rec`` features (square weight applied to both).
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_rec, n_rec)
                    )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    self.h.value = jax.nn.tanh(
                        braintrace.matmul(x_row, self.w.value)
                        + braintrace.matmul(self.h.value, self.w.value)
                    )
                    return self.h.value

                outs = brainstate.transform.for_loop(substep, jnp.arange(loops))
                return outs[-1]

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def snn_scan_rnn(n_rec: int = 4, loops: int = 40, decay: float = 0.9,
                 seed: int = 0) -> ModelSpec:
    """Leaky unit whose per-step update runs ``loops`` inner sub-steps
    ``h <- decay * h + tanh(matmul(x, w))`` in a ``for_loop``.

    The body's hidden-to-hidden path is the plain elementwise leak, so the
    per-substep Jacobian is exactly ``decay * I`` and structured scan
    descent (Phase 4) is exact: descended == unrolled == BPTT, including
    under chunked (online) gradient accumulation. This is the flagship
    diagonal-recurrence model for the descent oracle; with the default
    ``scan_unroll_limit`` a ``loops=40`` instance descends while ``loops=8``
    can be unrolled, so one factory serves both compile paths.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_rec, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    self.h.value = decay * self.h.value + jax.nn.tanh(
                        braintrace.matmul(x_row, self.w.value))
                    return self.h.value

                outs = brainstate.transform.for_loop(
                    substep, jnp.arange(loops))
                return outs[-1]

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),),
                     plain_param_keys=())


def snn_scan_two_state_rnn(n_rec: int = 3, loops: int = 40,
                           seed: int = 0) -> ModelSpec:
    """Two coupled hidden states (v, a) updated inside a ``for_loop`` —
    the descended analogue of :func:`two_state_rnn`.

    ``v <- 0.9 v + matmul(x, w) - 0.1 a``; ``a <- 0.95 a + v`` per sub-step.
    The compiler groups (v, a) into ONE descended HiddenGroup with
    ``num_state == 2``; the per-substep Jacobian is a per-position 2x2
    block, exercising the trailing learning-signal axis through the
    substep fold. Diagonal-recurrence class: D_RTRL descent is exact.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_rec, n_rec))
                self.v = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.a = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    v, a = self.v.value, self.a.value
                    self.v.value = 0.9 * v + braintrace.matmul(
                        x_row, self.w.value) - 0.1 * a
                    self.a.value = 0.95 * a + v
                    return self.v.value

                outs = brainstate.transform.for_loop(
                    substep, jnp.arange(loops))
                return outs[-1]

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),),
                     plain_param_keys=())


def _ring_csr(n_rec: int, offsets: Tuple[int, ...]):
    """The CSR matrix connecting ``q -> (q + off) % n_rec`` for each offset.

    Shared by the ring fixtures below so their position graphs are the *same*
    graph: a test that reads a neighbourhood size off one of them and an
    expected ``K(n)`` off the other is comparing like with like.

    Returns
    -------
    tuple
        ``(csr, nnz)`` — the matrix and its stored-entry count, which is the
        shape of the ETP data vector.
    """
    import brainevent

    dense_mask = np.zeros((n_rec, n_rec), dtype='float32')
    for q in range(n_rec):
        for off in offsets:
            dense_mask[q, (q + off) % n_rec] = 1.0
    return brainevent.CSR.fromdense(jnp.asarray(dense_mask)), int(dense_mask.sum())


def sparse_ring_rnn(
    n_in: int = 3, n_rec: int = 6, offsets: Tuple[int, ...] = (0, 1), seed: int = 0
) -> ModelSpec:
    """Tanh RNN whose recurrence is a **fixed sparse ring**, via ``sparse_matmul``.

    ``h^t = tanh(x^t @ win + sparse_matmul(h^{t-1}, w, sparse_mat=CSR))`` where
    the CSR pattern connects ``q -> (q + off) % n_rec`` for each ``off`` in
    ``offsets``. This is the reference model for the ``sparse_n`` recurrence
    scope: unlike a dense recurrent weight — whose position graph has diameter
    1, so ``n = 2`` already saturates — a ring of ``n_rec`` units has diameter
    ``n_rec - 1``, so the SnAp neighbourhood grows one position per order and
    ``K(n) == min(n, n_rec)`` exactly.

    The default ``offsets=(0, 1)`` keeps the **self** edge. Without it (a pure
    cycle) position ``p``'s hidden state does not depend on its own previous
    value at all, so the per-position block of the recurrent Jacobian is
    identically zero and ``recurrence_scope='diagonal'`` and ``'coupled'``
    produce *bit-identical* gradients — which would make every negative control
    that separates them vacuous on this model. The self edge is also the more
    realistic recurrent unit. It does not change ``K(n)``: closing ``I | shift``
    still reaches exactly one further position per order.

    Structural properties the acceptance suite pins:

    * one hidden group, ``varshape == (n_rec,)``, ``num_state == 1`` — the
      ``S = 1, K > 1`` configuration that exercises every ``num_state == 1``
      shortcut in the engine under a widened trace;
    * the relation's primitive is ``etp_sp_mv``, whose D-RTRL trace is
      ``nnz``-shaped rather than position-shaped, so it also pins that the
      widening is transparent to a primitive with a non-trivial anchor map;
    * the ``y -> hidden`` tail is elementwise (``add`` then ``tanh``), so a
      saturated within-group SnAp is full RTRL and must equal BPTT.

    Parameters
    ----------
    n_in : int, optional
        Input dimension.
    n_rec : int, optional
        Number of recurrent units (the ring length, hence the diameter).
    offsets : tuple of int, optional
        Ring offsets present in the sparse pattern. ``(1,)`` gives the pure
        cycle; adding offsets shortens the diameter.
    seed : int, optional
        Seed for the deterministic weight draw.

    Returns
    -------
    ModelSpec
        Spec whose ETP parameter is the sparse data vector ``w`` (shape
        ``(nnz,)``) and whose plain parameter is the input projection ``win``.
    """
    csr, nnz = _ring_csr(n_rec, offsets)

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.6 * brainstate.random.randn(nnz))
                    self.win = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                rec = braintrace.sparse_matmul(self.h.value, self.w.value, sparse_mat=csr)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def rolled_tail_rnn(
    n_in: int = 3, n_rec: int = 5, roll: int = 1, seed: int = 0
) -> ModelSpec:
    """Dense tanh RNN whose ``y -> hidden`` tail **relabels positions** (F-31).

    ``h^t = tanh(x @ win + roll(matmul(h^{t-1}, w), roll))``. The mixing
    primitive is the ordinary dense ``etp_mm``/``etp_mv``, whose position graph
    has diameter 1, so ``sparse_n`` saturates at ``n = 2`` and a *saturated*
    within-group rule is full within-group RTRL. What the roll breaks is the
    premise underneath the whole ``recurrence_scope`` axis: the trace indexes
    hidden units by position, and here the position that a mixing output lands
    on is not the position it was computed for.

    ``roll=0`` gives the control — the same model with a position-preserving
    tail — which is what makes the comparison legible: at ``roll=0`` saturation
    equals BPTT to round-off, and at ``roll=1`` it does not, while the model is
    otherwise identical. Use the pair, never the rolled model alone.

    The position analysis detects the tail (the ``slice`` equations ``roll``
    lowers to are not position-preserving) and widens to all-to-all with a
    ``SNAP_PATTERN_CONSERVATIVE`` diagnostic, so the shortfall is warned about
    rather than silent. See "Notes on F-31" in
    ``docs/specs/2026-07-25-known-limitations.md``.

    Parameters
    ----------
    n_in : int, optional
        Input dimension.
    n_rec : int, optional
        Number of recurrent units.
    roll : int, optional
        Positions to roll the recurrent term by. ``0`` disables the relabelling
        and yields the control model.
    seed : int, optional
        Seed for the deterministic weight draw.

    Returns
    -------
    ModelSpec
        Spec whose ETP parameter is the recurrent matrix ``w``.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.6 * brainstate.random.randn(n_rec, n_rec))
                    self.win = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec))
                self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                rec = braintrace.matmul(self.h.value, self.w.value)
                if roll:
                    rec = jnp.roll(rec, roll)
                self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def sparse_ring_two_state_rnn(
    n_in: int = 3, n_rec: int = 5, offsets: Tuple[int, ...] = (0, 1), seed: int = 0
) -> ModelSpec:
    """:func:`sparse_ring_rnn` with a second, coupled hidden state.

    ``v^t = tanh(x @ win + sparse_matmul(v^{t-1}, w) - 0.1 a^{t-1})`` and
    ``a^t = 0.95 a^{t-1} + v^{t-1}``. The compiler groups ``(v, a)`` into one
    HiddenGroup with ``num_state == 2``, so the widened trace axis is
    ``M = K * 2`` — the ``S > 1, K > 1`` configuration. The adjacency analysis
    still sees exactly one mixing equation (on ``v``); the ``a`` coupling is
    hand-written arithmetic and contributes no position mixing, which is why
    the pattern stays the precise ring rather than falling back to conservative.

    Parameters
    ----------
    n_in, n_rec, offsets, seed
        As in :func:`sparse_ring_rnn`.

    Returns
    -------
    ModelSpec
        Spec whose ETP parameter is the sparse data vector ``w``.
    """
    csr, nnz = _ring_csr(n_rec, offsets)

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.6 * brainstate.random.randn(nnz))
                    self.win = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec))
                self.v = brainstate.HiddenState(jnp.zeros((n_rec,)))
                self.a = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                v, a = self.v.value, self.a.value
                rec = braintrace.sparse_matmul(v, self.w.value, sparse_mat=csr)
                self.v.value = jax.nn.tanh(x @ self.win.value + rec - 0.1 * a)
                self.a.value = 0.95 * a + v
                return self.v.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def while_settle_rnn(
    n_in: int = 3, n_rec: int = 4, k: int = 3, decay: float = 0.8, seed: int = 0
) -> ModelSpec:
    """Leaky drive followed by a **weight-free** ``lax.while_loop`` settle.

    ``pre = matmul(x, win) + decay * h_prev`` (ETP input weight, plain leak),
    then ``k`` settle iterations ``h <- h + 0.5 * tanh(pre - h)`` inside a
    ``lax.while_loop`` starting from ``h_prev``. The loop consumes only
    weight-*derived* values, so the compiler keeps it as an opaque forward
    node (Phase 3 ``while_hidden='opaque-fwd'``): hidden Jacobians are
    extracted in forward mode and the perturbation pass detaches the loop
    inputs. :func:`while_settle_twin_rnn` with the same ``seed`` builds the
    mathematically identical model with the loop hand-composed, so the pair
    isolates the while-specific machinery.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.win = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_in, n_rec)
                    )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                h_prev = self.h.value
                pre = braintrace.matmul(x.reshape(1, -1), self.win.value) + decay * h_prev

                def body(s):
                    i, h = s
                    return i + 1, h + 0.5 * jnp.tanh(pre - h)

                _, h_new = jax.lax.while_loop(lambda s: s[0] < k, body, (0, h_prev))
                self.h.value = h_new
                return h_new

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('win',),), plain_param_keys=())


def while_settle_twin_rnn(
    n_in: int = 3, n_rec: int = 4, k: int = 3, decay: float = 0.8, seed: int = 0
) -> ModelSpec:
    """Hand-composed twin of :func:`while_settle_rnn` (same ``seed`` gives
    identical weights): the fixed-trip-count while is replaced by its
    ``k``-fold composition, so reverse-mode oracles (BPTT) apply."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.win = brainstate.ParamState(
                        0.1 * brainstate.random.randn(n_in, n_rec)
                    )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                h_prev = self.h.value
                pre = braintrace.matmul(x.reshape(1, -1), self.win.value) + decay * h_prev
                h = h_prev
                for _ in range(k):
                    h = h + 0.5 * jnp.tanh(pre - h)
                self.h.value = h
                return h

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('win',),), plain_param_keys=())


# ---------------------------------------------------------------------------
# P4 specs: fixtures whose *shape* is the point.
#
# Each of these exists because a P4 acceptance criterion is vacuous without it,
# and each carries the measurement that made it necessary in its docstring. They
# are deliberately small: `nonzero_init_rnn` is enumerated over 16 sign patterns,
# so every element costs 16 algorithm constructions.
# ---------------------------------------------------------------------------


def nonzero_init_rnn(n_rec: int = 2, h0: float = 0.4, seed: int = 0) -> ModelSpec:
    """Minimal dense tanh RNN with a **non-zero initial hidden state**.

    ``h^t = tanh(x^t + matmul(h^{t-1}, w))`` — one square ETP weight, no plain
    parameters, so ``x`` must carry ``n_rec`` features. This is the reference
    model for ``trace_factorization='random_projection'`` (UORO), where the pin
    is that the rolled hidden-to-hidden Jacobian is the *full* within-group one.

    The non-zero ``h0`` is not cosmetic, and it is the reason this spec exists
    rather than reusing :func:`tanh_rnn`. With ``h^0 = 0`` the recurrent weight's
    first instantaneous term is identically zero, so the transition is applied to
    a zero influence and rolling the full Jacobian versus its block diagonal
    become indistinguishable. Measured at ``n_rec = 2``, ``T = 3``, one-step
    windows, exhaustive over the two draws that reach a boundary (16 runs),
    deviation of the enumeration mean from BPTT:

    | initial hidden | full ``D`` | block-diagonal ``D`` |
    |---|---|---|
    | ``h0 = 0``     | 1.0e-16 | 3.3e-16 |   <- pin passes for the wrong rule
    | ``h0 = 0.4``   | 2.1e-16 | 2.6e-04 |   <- pin discriminates

    ``T >= 3`` matters for the same reason: at ``T <= 2`` with one-step windows
    the boundary trace holds only instantaneous terms and ``D`` never enters.

    ``init_state``/``reset_state`` restore ``h0``, so a re-initialized model
    repeats a run bit-for-bit -- the other fixtures in this module set their
    hidden value in ``__init__`` only, which means ``init_all_states`` leaves a
    *used* model wherever the last step left it.

    Parameters
    ----------
    n_rec : int, optional
        Number of recurrent units. Also the required input width.
    h0 : float, optional
        The constant initial hidden value. Must be non-zero for the fixture to
        do its job; ``0.0`` is accepted so tests can build the negative control.
    seed : int, optional
        Seed for the deterministic weight draw.

    Returns
    -------
    ModelSpec
        Spec whose only parameter is the ETP recurrent weight ``w``.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.6 * brainstate.random.randn(n_rec, n_rec))
                self.h = brainstate.HiddenState(jnp.full((1, n_rec), h0))

            def init_state(self, batch_size=None, **kwargs):
                size = (n_rec,) if batch_size is None else (batch_size, n_rec)
                self.h.value = jnp.full(size, h0)

            def reset_state(self, batch_size=None, **kwargs):
                self.init_state(batch_size, **kwargs)

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def unit_weight_rnn(n_in: int = 3, n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Two coupled hidden states carrying **different physical units**.

    ``v`` is in mV and ``a`` is in nA, and the compiler groups them into one
    HiddenGroup with ``num_state == 2``, so ``concat_hidden`` has to strip two
    *different* units into one mantissa array. The single ETP parameter ``w`` is
    in mV.

    This is the fixture for every claim of the form "the scalar is applied to
    mantissas only". A normaliser computed on a concatenated hidden vector whose
    entries are mV and nA has no meaningful unit, so an implementation that
    tries to keep units through ``rho`` either raises a
    ``brainunit`` dimension error here or silently compares mV against nA. A
    single-unit model cannot tell the two apart.

    Parameters
    ----------
    n_in : int, optional
        Input dimension (dimensionless input).
    n_rec : int, optional
        Units per hidden state.
    seed : int, optional
        Seed for the deterministic weight draw.

    Returns
    -------
    ModelSpec
        Spec whose only parameter is the ETP input weight ``w`` (in mV).
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec) * u.mV)
                self.v = brainstate.HiddenState(jnp.zeros((1, n_rec)) * u.mV)
                self.a = brainstate.HiddenState(jnp.zeros((1, n_rec)) * u.nA)

            def update(self, x):
                v, a = self.v.value, self.a.value
                self.v.value = (0.9 * v
                                + braintrace.matmul(x, self.w.value)
                                - a * (1.0 * u.mV / u.nA))
                self.a.value = 0.95 * a + v * (0.1 * u.nA / u.mV)
                return self.v.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def plain_and_etp_rnn(
    n_in: int = 3, n_rec: int = 4, n_out: int = 2, seed: int = 0
) -> ModelSpec:
    """Tanh RNN with one ETP weight and **two kinds of plain weight**.

    ``h^t = tanh(x^t @ win + matmul(h^{t-1}, w))``, output ``h^t @ wout``:

    * ``w`` — ETP recurrent weight; its cross-window credit already flows,
      through the eligibility trace;
    * ``win`` — plain, and reaches every *future* loss through the recurrence, so
      a finite window truncates its credit. This is the parameter
      ``learning_signal='bootstrapped'`` exists to repair;
    * ``wout`` — plain, but reaches only the loss at its own step, so its
      windowed gradient already equals the full-sequence one.

    Both plain kinds are needed. ``wout`` is the control that turns "DNI changed
    the plain gradients" into "DNI changed exactly the plain gradients that were
    truncated": a synthesiser that leaks credit into ``wout`` is wrong, and a
    fixture with only ``win`` cannot see it. Measured, ``T = 4``, ``n_rec = 4``,
    two-step windows, max-abs gradient:

    | key | BPTT | windowed (D_RTRL, multi-step) |
    |---|---|---|
    | ``w``    | 1.437 | 1.437 (exact -- the trace carries it) |
    | ``win``  | 4.289 | 3.175 (truncated) |
    | ``wout`` | 4.060 | 4.060 (nothing to truncate) |

    Note that under ``vjp_method='single-step'`` *both* plain keys are identically
    zero, not merely truncated (F-33), so this fixture belongs to the multi-step
    windowed path.

    Parameters
    ----------
    n_in, n_rec, n_out : int, optional
        Input, recurrent and output widths.
    seed : int, optional
        Seed for the deterministic weight draws.

    Returns
    -------
    ModelSpec
        Spec with ETP ``w`` and plain ``win``, ``wout``.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_rec, n_rec))
                    self.win = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec))
                    self.wout = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_rec, n_out))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x @ self.win.value
                    + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value @ self.wout.value

        return Net()

    return ModelSpec(
        factory=factory,
        etp_param_keys=(('w',),),
        plain_param_keys=(('win',), ('wout',)),
    )


def delayed_reward_rnn(
    n_in: int = 2, n_rec: int = 8, leak: float = 0.95, seed: int = 0
) -> ModelSpec:
    """Long-memory leaky RNN for a **delayed-reward** task.

    ``h^t = leak * h^{t-1} + (1 - leak) * tanh(x^t @ win + matmul(h^{t-1}, w))``,
    scalar output ``h^t @ wout``. The near-unit leak is the point: credit for an
    input at step 0 survives to step 20 (``0.95^20 = 0.36``), so a reward
    delivered at the end of the sequence has to be attributed across many
    windows.

    The ``(1 - leak)`` factor is a **convex** combination, not decoration. Without
    it the state is a plain accumulator bounded only by ``1 / (1 - leak) = 20``,
    and every downstream quantity inherits that scale: outputs reach ``O(10)``,
    squared errors ``O(100)``, and the hidden cotangents that
    :func:`~braintrace.train_synthetic_gradient` regresses against reach
    ``O(1e5)``. Measured on the unscaled form at ``T=24``, the synthesiser's
    regression loss started at ``1.7e5`` and diverged to ``nan`` under its
    built-in SGD, and plain-SGD training of the model itself diverged at every
    learning rate down to ``2e-3``. With the factor, ``|h| <= 1`` for all ``T``.
    The credit span is untouched -- it comes from the ``leak * h`` term, whose
    Jacobian contribution is still ``leak * I``.

    This is the fixture for the ``bootstrapped`` end-to-end criterion, which a
    bandit cannot serve -- with no temporal credit to carry there is nothing for
    a synthetic gradient to supply, and the control runs would tie. The task
    itself (cue placement, reward definition, held-out sequence) lives in the
    test, because the same model serves the modulatory smoke test with a
    different reward.

    Parameters
    ----------
    n_in : int, optional
        Input width. The task uses one cue channel and one distractor.
    n_rec : int, optional
        Recurrent width.
    leak : float, optional
        Carry coefficient. Values near 1 lengthen the credit span; ``leak`` also
        makes the hidden-to-hidden Jacobian well-conditioned, so BPTT on the
        same model is a stable reference.
    seed : int, optional
        Seed for the deterministic weight draws.

    Returns
    -------
    ModelSpec
        Spec with ETP ``w`` and plain ``win``, ``wout``.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.w = brainstate.ParamState(
                        0.2 * brainstate.random.randn(n_rec, n_rec))
                    self.win = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_in, n_rec))
                    self.wout = brainstate.ParamState(
                        0.5 * brainstate.random.randn(n_rec, 1))
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                self.h.value = leak * self.h.value + (1.0 - leak) * jax.nn.tanh(
                    x @ self.win.value
                    + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value @ self.wout.value

        return Net()

    return ModelSpec(
        factory=factory,
        etp_param_keys=(('w',),),
        plain_param_keys=(('win',), ('wout',)),
    )


def two_island_rnn(n_in: int = 3, n_rec: int = 3, seed: int = 0) -> ModelSpec:
    """Two **disconnected** recurrent subnetworks -- the only multi-group fixture.

    ``ha`` and ``hb`` each carry their own ETP weight (``wa``, ``wb``) and never
    read each other; only the loss sees both, via ``ha + hb``.

    This fixture exists because of a property of ``recurrence_scope='coupled'``
    that is easy to assume away: hidden grouping follows the *transition*, so
    coupled scope merges every set of mutually reachable hidden states into a
    single group. Every other spec in this module therefore compiles to exactly
    **one** group under coupled, including :func:`stacked_tanh_rnn` (whose two
    layers are joined by the plain inter-layer projection) and
    :func:`two_state_rnn` (whose ``v`` and ``a`` are joined by construction).
    Measured, ``stacked_tanh_rnn(n_in=4, n_rec=4)``:

    ==================  ============================
    ``recurrence_scope``  compiled hidden groups
    ==================  ============================
    ``'diagonal'``        2: ``[('h1',)]``, ``[('h2',)]``
    ``'coupled'``         1: ``[('h1',), ('h2',)]``
    ==================  ============================

    So a per-group carrier claim ("one hidden factor per group, one parameter
    factor per group-path pair") cannot be pinned on any pre-existing spec at
    coupled scope -- the group count is 1 and the two clauses are
    indistinguishable. Severing the two halves is what separates them.

    Parameters
    ----------
    n_in : int, optional
        Input width. The drive is *added* to each island's recurrent term rather
        than projected, so this must equal ``n_rec`` (or 1, to broadcast); there
        is deliberately no input weight to keep both islands purely ETP.
    n_rec : int, optional
        Width of each island; the compiled groups are ``n_rec`` wide, not
        ``2 * n_rec``.
    seed : int, optional
        Seed for the deterministic weight draws.

    Returns
    -------
    ModelSpec
        Spec with ETP ``wa`` and ``wb``, no plain parameters.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(seed):
                    self.wa = brainstate.ParamState(
                        0.2 * brainstate.random.randn(n_rec, n_rec))
                    self.wb = brainstate.ParamState(
                        0.2 * brainstate.random.randn(n_rec, n_rec))
                self.ha = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.hb = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def init_state(self, batch_size=None, **kwargs):
                shape = (batch_size or 1, n_rec)
                self.ha = brainstate.HiddenState(jnp.zeros(shape))
                self.hb = brainstate.HiddenState(jnp.zeros(shape))

            def update(self, x):
                self.ha.value = jax.nn.tanh(
                    x + braintrace.matmul(self.ha.value, self.wa.value))
                self.hb.value = jax.nn.tanh(
                    x + braintrace.matmul(self.hb.value, self.wb.value))
                return self.ha.value + self.hb.value

        return Net()

    return ModelSpec(
        factory=factory,
        etp_param_keys=(('wa',), ('wb',)),
        plain_param_keys=(),
    )


# ---------------------------------------------------------------------------
# SNN specs: the realistic-model end of the zoo.
#
# These wrap the layer classes in ``braintrace/_testing/models.py`` for
# oracle use. Two things have to be fixed at this boundary:
#
# * F-24 -- those constructors call unseeded ``braintools.init.*``, which draws
#   from the global ``brainstate.random`` stream, so ``factory()`` returns a
#   different model on every call and a BPTT-vs-online comparison would compare
#   two different networks. Each factory re-seeds before constructing.
# * F-25 -- at unit input scale the neurons never reach threshold, so the loss
#   and the gradient are identically zero. Each spec records the scale that
#   makes it live; ``oracle_models_test.py`` asserts both properties.
#
# The live input-scale window is bounded on **both** sides, which is the part of
# F-25 that is easy to miss. Too little drive and the neuron never crosses
# threshold; too much and the surrogate derivative saturates, so the gradient is
# exactly zero again *while the network keeps spiking*. Measured for
# ``ALIF_Delta`` at n_in=4, n_rec=5, T=6:
#
#     scale  spike_rate  |g_bptt|
#      0.05     0.00      0.0        <- under threshold
#      0.20     0.17      3.0e+00
#      1.00     0.53      9.3e+00    <- chosen
#      2.00     0.60      0.0        <- saturated surrogate, still spiking
#     20.00     0.60      0.0
#
# So spike rate is not a proxy for liveness, and the per-spec scale below is not
# a free parameter: conductance-based (ExpCu/ExpCo) layers need a large scale,
# while delta layers inject straight into mV and need a small one.
# ---------------------------------------------------------------------------

_SNN_SEED = 7
_SNN_SCALE = 20.0        # conductance-based layers
_SNN_SCALE_DELTA = 1.0   # ALIF + delta synapse: saturates above ~2.0


def _snn_spec(cls, n_in, n_rec, seed, scale=_SNN_SCALE, **kwargs) -> ModelSpec:
    """Wrap an SNN layer class as a deterministic, live ``ModelSpec``."""

    def factory():
        brainstate.random.seed(seed)
        return cls(n_in, n_rec, **kwargs)

    return ModelSpec(
        factory=factory,
        etp_param_keys=(),   # discovered by the compiler; not asserted per-spec
        plain_param_keys=(),
        input_scale=scale,
        batched_input=True,
    )


def snn_if_delta(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """IF neuron, delta synapse. Single hidden state (``num_state == 1``)."""
    from braintrace._testing.models import IF_Delta_Dense_Layer
    return _snn_spec(IF_Delta_Dense_Layer, n_in, n_rec, seed)


def snn_alif_delta(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF neuron, delta synapse. Membrane + adaptation (``num_state == 2``).

    Uses the smaller delta scale: the synapse injects directly in mV, so the
    conductance-model scale saturates the surrogate derivative and drives the
    gradient to exactly zero while the network still spikes. See the F-25 note
    above this block.
    """
    from braintrace._testing.models import ALIF_Delta_Dense_Layer
    return _snn_spec(ALIF_Delta_Dense_Layer, n_in, n_rec, seed,
                     scale=_SNN_SCALE_DELTA)


def snn_lif_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF neuron, exponential current synapse. Two timescales: tau_mem, tau_syn."""
    from braintrace._testing.models import LIF_ExpCu_Dense_Layer
    return _snn_spec(LIF_ExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_alif_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF + exponential current synapse. Three timescales, ``num_state == 3``."""
    from braintrace._testing.models import ALIF_ExpCu_Dense_Layer
    return _snn_spec(ALIF_ExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_lif_std_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF + short-term depression. Adds tau_std as a further timescale."""
    from braintrace._testing.models import LIF_STDExpCu_Dense_Layer
    return _snn_spec(LIF_STDExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_lif_stp_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF + short-term plasticity. Adds tau_f and tau_d."""
    from braintrace._testing.models import LIF_STPExpCu_Dense_Layer
    return _snn_spec(LIF_STPExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_alif_expco_ei(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF with an excitatory/inhibitory population split and conductance
    synapses. The heterogeneous-population case: separate E and I projections
    produce several ETP relations feeding one hidden group."""
    from braintrace._testing.models import ALIF_ExpCo_Dense_Layer
    return _snn_spec(ALIF_ExpCo_Dense_Layer, n_in, n_rec, seed)


def snn_lif_expcu_heterogeneous(
    n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED
) -> ModelSpec:
    """LIF whose membrane time constant differs per neuron.

    The heterogeneous-leak case: ``tau_mem`` is a length-``n_rec`` vector, so no
    single global leak exists for the transition to factor out.
    """
    from braintrace._testing.models import LIF_ExpCu_Dense_Layer
    tau_mem = jnp.linspace(3.0, 12.0, n_rec) * u.ms
    return _snn_spec(LIF_ExpCu_Dense_Layer, n_in, n_rec, seed, tau_mem=tau_mem)


def snn_alif_expcu_heterogeneous(
    n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED
) -> ModelSpec:
    """ALIF with per-neuron membrane *and* adaptation time constants."""
    from braintrace._testing.models import ALIF_ExpCu_Dense_Layer
    return _snn_spec(
        ALIF_ExpCu_Dense_Layer, n_in, n_rec, seed,
        tau_mem=jnp.linspace(3.0, 12.0, n_rec) * u.ms,
        tau_a=jnp.linspace(60.0, 150.0, n_rec) * u.ms,
    )


SNN_SPECS = {
    'if_delta': snn_if_delta,
    'alif_delta': snn_alif_delta,
    'lif_expcu': snn_lif_expcu,
    'alif_expcu': snn_alif_expcu,
    'lif_std_expcu': snn_lif_std_expcu,
    'lif_stp_expcu': snn_lif_stp_expcu,
    'alif_expco_ei': snn_alif_expco_ei,
    'lif_expcu_heterogeneous': snn_lif_expcu_heterogeneous,
    'alif_expcu_heterogeneous': snn_alif_expcu_heterogeneous,
}
