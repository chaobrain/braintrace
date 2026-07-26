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

"""``trace_factorization='random_projection'`` — the rank-1 unbiased trace (UORO).

The engine behind :class:`~braintrace.UORO`. See that class for the user-facing
description; this module documents the representation and the update.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintrace._compiler import (
    ControlFlowPolicy,
    DEFAULT_MAX_JACOBIAN_ELEMENTS,
    HiddenGroup,
    HiddenParamOpRelation,
)
from braintrace._misc import etrace_df_key
from braintrace._op import etp_elemwise_p, is_batched_primitive
from braintrace._typing import (
    Hid2WeightJacobian,
    HiddenGroupJacobian,
    Path,
    PyTree,
)
from ._common import _route_grads_by_path, _update_dict
from .axes import ETraceConfig
from .param_dim_vjp import (
    reduce_param_batch_axes,
    relation_instant_term,
    relation_solve_to_param,
    relation_weights_dict,
)
from .vjp_base import ETraceVjpAlgorithm

__all__ = [
    'RandomProjectionVjpAlgorithm',
]

#: Reserved keys in the eligibility-trace carry. The carry is a plain dict so it
#: flattens as a pytree, and every sub-dict is keyed homogeneously (ints for
#: groups, ``(group index, path)`` tuples for parameters) because JAX sorts dict
#: keys when flattening.
_S_TILDE = 's_tilde'
_THETA_TILDE = 'theta_tilde'
_KEY = 'key'
_STEP = 'step'


def _tree_sq_norm(tree: PyTree) -> jax.Array:
    """Sum of squares over every leaf's **mantissa**.

    Units are stripped rather than carried. A group's hidden vector may
    concatenate states in different units (mV and nA in the same group is a
    supported model), so a norm over it has no meaningful unit, and the
    normalisers are dimensionless scalars by contract.
    """
    total = jnp.zeros((), dtype=jnp.float32)
    for leaf in jax.tree.leaves(tree):
        mantissa = u.get_mantissa(leaf)
        total = total + jnp.sum(
            jnp.square(mantissa.astype(jnp.promote_types(mantissa.dtype, jnp.float32)))
        )
    return total


def _tree_norm(tree: PyTree) -> jax.Array:
    """Euclidean norm over every leaf's mantissa."""
    return jnp.sqrt(_tree_sq_norm(tree))


def _at_precision_of(value: Any, template: Any) -> Any:
    """*value* narrowed back to *template*'s dtype, unit-preserving.

    ``_tree_sq_norm`` accumulates in at least float32 deliberately — a sum of
    squares in float16 underflows for state magnitudes below about ``1e-3`` — so
    the normalisers come back wider than a half-precision model's factors, and
    scaling a factor by one promotes it. The rank-1 factors are ``scan`` carries,
    and ``jax.lax.scan`` rejects a carry whose output dtype differs from its
    input: not a silent downcast but a hard ``carry input and carry output must
    have equal types`` on the first ``MultiStepData`` call. Reducing in the wider
    dtype and narrowing the *result* keeps both the accuracy and the contract.

    Parameters
    ----------
    value : jax.Array or brainunit.Quantity
        The freshly combined factor.
    template : jax.Array or brainunit.Quantity
        The carry slot *value* has to fit, whose dtype is authoritative.

    Returns
    -------
    jax.Array or brainunit.Quantity
        *value* at *template*'s dtype.
    """
    return u.math.astype(value, jax.dtypes.result_type(u.get_mantissa(template)))


def _scale_tree(tree: PyTree, scale: jax.Array) -> PyTree:
    """Multiply every leaf by a dimensionless scalar, unit-safely."""
    return jax.tree.map(lambda a: a * scale, tree, is_leaf=u.math.is_quantity)


def _apply_full_transition(jac: jax.Array, s_tilde: jax.Array) -> jax.Array:
    r"""``D_full @ s_tilde`` for a full ``(*V, S, *V, S)`` Jacobian.

    ``jac[p, a, q, b] == d f(h)[p, a] / d h[q, b]``, so the contraction pairs
    ``jac``'s *trailing* ``s_tilde.ndim`` axes with all of ``s_tilde``'s.

    Parameters
    ----------
    jac : jax.Array
        The group's full hidden-to-hidden Jacobian.
    s_tilde : jax.Array
        The hidden-side factor, shaped ``(*varshape, num_state)``.

    Returns
    -------
    jax.Array
        Same shape as *s_tilde*.
    """
    n = s_tilde.ndim
    if jac.ndim != 2 * n:
        raise ValueError(
            f'random_projection needs the full hidden-to-hidden Jacobian with '
            f'{2 * n} axes for a factor of {n} axes, but got a Jacobian with '
            f'{jac.ndim} axes (shape {jnp.shape(jac)}). This is what the '
            f'executor\'s full_jacobian flag selects; a block-diagonal Jacobian '
            f'cannot be rolled by a rank-1 carrier.'
        )
    return u.math.tensordot(jac, s_tilde, axes=n)


class RandomProjectionVjpAlgorithm(ETraceVjpAlgorithm):
    r"""Rank-1 random-projection eligibility trace — the UORO engine.

    Per hidden group ``g`` the influence of parameter ``j`` on hidden unit
    ``u = (position, state)`` is carried by **one rank-1 pair**

    .. math::

        \tilde\varepsilon_g[j, u] \approx \tilde s_g[u] \, \tilde\theta_g[j]

    with :math:`\tilde s_g` shaped ``(*varshape, num_state)`` and
    :math:`\tilde\theta_g` parameter-shaped — no hidden axis, no trailing state
    axis. One :math:`\tilde s` per group; one :math:`\tilde\theta` per
    ``(group, ParamState path)``, so a weight consumed by two relations of one
    group keeps a single parameter factor and the two projections are summed into
    it.

    With :math:`\nu_g` a Rademacher draw of :math:`\tilde s_g`'s shape and
    :math:`J_f` the instantaneous term, the step is

    .. math::

        \mathrm{proj}_g &= \nu_g^\top J_f \\
        \rho_0 &= \sqrt{(\|\tilde\theta_g\| + \epsilon) /
                        (\|D_g \tilde s_g\| + \epsilon)} \\
        \rho_1 &= \sqrt{(\|\mathrm{proj}_g\| + \epsilon) /
                        (\|\nu_g\| + \epsilon)} \\
        \tilde s_g &\leftarrow \rho_0 (D_g \tilde s_g) + \rho_1 \nu_g \\
        \tilde\theta_g &\leftarrow \tilde\theta_g / \rho_0 +
                                   \mathrm{proj}_g / \rho_1

    and the gradient contributed at a window boundary is
    :math:`\sum_g (\text{signal}_g \cdot \tilde s_g)\, \tilde\theta_g` — a scalar
    per group times a parameter-shaped array.

    **Unbiased for what.** Conditionally on earlier draws, the updated outer
    product is ``a b^T + c d^T + (rho0/rho1) a d^T + (rho1/rho0) c b^T`` with
    ``a = D s_tilde``, ``b = theta_tilde``, ``c = nu``, ``d = nu^T J_f``. The
    first term is ``D eps_tilde`` (``rho0`` cancels), the second is
    ``nu nu^T J_f``, and the two cross terms are **odd** in the current draw while
    ``rho1`` is even, so any negation-symmetric draw law with
    ``E[nu nu^T] == I`` gives ``E[eps_tilde_new] == D eps_tilde + J_f`` exactly.
    Induction over time then gives an unbiased estimate of **the exact
    within-group influence recursion that the compiled transition defines**.

    That is the whole claim, and it is narrower than "unbiased". Three
    approximations are untouched:

    1. **cross-group coupling** — the compiler splits hidden states into groups
       and drops inter-group terms; one factor pair per group cannot carry them;
    2. **the instantaneous term's tail** — ``df`` comes from a single all-ones
       JVP of the ``y -> hidden`` map, exact only for a position-preserving
       elementwise tail (finding F-31);
    3. **each primitive's own solve regime** — the projector is built from the
       framework's own per-primitive rules, so it inherits them exactly.

    On a single-group model with a position-preserving elementwise tail — the
    class where that recursion is itself exact — this is therefore unbiased for
    BPTT. Elsewhere it is unbiased for the block-local recursion and biased
    against BPTT, like every other coordinate in this repository. It is *not*
    restricted to that class.

    **Memory, honestly.** The carrier is ``|theta| + B * P * S`` elements against
    ``B * |theta| * S`` for the anchored per-parameter trace: on
    ``tanh_rnn(3, 4)`` that is 20 against 16. UORO is **not** a memory win over
    the anchored trace — it is a *bias* win at comparable carrier size, and a
    memory win against saturated SnAp-n (64) or the full influence matrix (64),
    which is what unbiasedness would otherwise cost. And ``O(|theta| + P S)``
    describes **carrier storage, not peak memory**: the full transition Jacobian
    is an ``O((P S)^2)`` transient per step (follow-up F-32 tracks a matrix-free
    ``D @ s`` product). The fused stepper keeps that transient per-step rather
    than stacked over the window.

    Parameters
    ----------
    model : brainstate.nn.Module
        The one-step model.
    name : str, optional
        Node name.
    vjp_method : str, optional
        ``'single-step'`` or ``'multi-step'``. Default ``'multi-step'``: the
        chunked finite-window path requires it, and it is the path the estimator
        is validated on.
    fast_solve : bool, optional
        Whether registered closed-form kernels may be used when building the
        projector. Default ``True``.
    control_flow : ControlFlowPolicy, optional
        Control-flow canonicalization policy.
    config : ETraceConfig, optional
        The learning-rule coordinate. Must have
        ``trace_factorization='random_projection'``.
    projection_key : int or jax.Array, optional
        Seed for the projection stream. Default ``42``. ``reset_state``
        re-derives the stream from it, so a reset run repeats bit-for-bit.
    projection_eps : float, optional
        Guard added to every norm before the ratio. Default ``1e-12``. At the
        first step both factors are zero, so both ratios are ``0/0``; with
        ``projection_eps=0`` the carrier is NaN at every window length. Because
        the guard perturbs the normalisers, the exactness pins carry a tolerance
        above float64 epsilon — see ``uoro_test.py``.
    random_feedback_key : jax.Array, optional
        Passed through to the base class for ``learning_signal='random_feedback'``.
    snap_max_jacobian_elements : int, optional
        Passed through; unused at ``recurrence_scope='coupled'``.

    See Also
    --------
    braintrace.UORO : the preset that fixes the coordinate.
    braintrace.SnAp : the deterministic scale whose saturating end this
        estimates.
    """

    __module__ = 'braintrace'

    #: The coordinate this engine implements.
    _default_config = ETraceConfig(
        trace_factorization='random_projection', recurrence_scope='coupled')

    _supports_scan_descent = False
    """A descended scan body is analysed with ``include_recurrent_mixing=False``,
    so a "full" Jacobian built there would be missing exactly the recurrent
    mixing this engine exists to roll. Refuse instead of degrading silently."""

    _scan_descent_refusal_reason = (
        "The refusal is a coordinate conflict, not a missing feature: a "
        "descended scan body is analysed with include_recurrent_mixing=False, "
        "so the transition available inside it is block-diagonal. Rolling a "
        "rank-1 estimate of a block-diagonal transition would converge onto the "
        "biased trace, which recurrence_scope='coupled' exists to avoid -- so "
        "the estimator would cost variance and buy nothing."
    )

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'multi-step',
        fast_solve: bool = True,
        control_flow: Optional[ControlFlowPolicy] = None,
        config: Optional[ETraceConfig] = None,
        projection_key: Any = 42,
        projection_eps: float = 1e-12,
        random_feedback_key: Optional[jax.Array] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ) -> None:
        super().__init__(model, name=name, vjp_method=vjp_method,
                         control_flow=control_flow, config=config,
                         random_feedback_key=random_feedback_key,
                         snap_max_jacobian_elements=snap_max_jacobian_elements)
        if self.config.trace_factorization != 'random_projection':
            raise ValueError(
                f'{type(self).__name__} is the random-projection trace engine, '
                f'but the config asks for '
                f'trace_factorization={self.config.trace_factorization!r}. Use '
                f'the engine matching the factorization, or '
                f'braintrace.compile(model, config, ...) which picks it for you.'
            )
        self.fast_solve = fast_solve
        self.projection_key = projection_key
        self.projection_eps = float(projection_eps)
        # Rejected rather than documented. At the first step both factors are zero
        # so both normalisers are `0/0`; a guard that is zero -- or one so small it
        # rounds to zero in the float32 the norms are accumulated in -- makes the
        # carrier NaN at *every* window length, and the NaN then propagates to
        # every gradient. There is no configuration in which that is the intended
        # behaviour, so it is an error at construction rather than a surprise at
        # the first backward pass.
        if not (self.projection_eps > 0.0
                and float(jnp.asarray(self.projection_eps, jnp.float32)) > 0.0):
            raise ValueError(
                f'projection_eps must be positive and remain positive in float32, '
                f'got {projection_eps!r}. It guards the `0/0` both normalisers hit '
                f'at the first step (and that a dead group hits at any step); '
                f'without it every gradient is NaN. The default, 1e-12, is a '
                f'reasonable choice.'
            )

    # ------------------------------------------------------------------ #
    # the projection stream
    # ------------------------------------------------------------------ #

    def _initial_projection_key(self) -> jax.Array:
        """The PRNG key the stream starts from.

        ``brainstate.random`` rather than ``jax.random`` per AGENTS.md rule 11,
        and an *explicit* carried key rather than the global counter: the stepper
        runs inside ``jax.custom_vjp`` under a scan, where a global draw is
        neither replayable across the fwd/bwd pair nor stable under
        ``reset_state``.
        """
        key = self.projection_key
        if isinstance(key, int):
            return jnp.asarray(brainstate.random.RandomState(key).split_key())
        return jnp.asarray(key)

    def _draw_projection(
        self,
        key: jax.Array,
        step: jax.Array,
        group_index: int,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> jax.Array:
        """Draw one group's projection vector. **Protected test seam.**

        The default is a Rademacher draw: ``±1`` with equal probability, which is
        negation-symmetric with ``E[nu nu^T] == I`` — the two properties the
        unbiasedness argument needs, and the reason an exhaustive enumeration over
        sign patterns is exact rather than merely convergent.

        An override must be **functional**. The stepper body is traced once and
        run under ``scan``, so an override that consumed a Python iterator would
        bake a single draw into every step; index a stacked table by *step*
        instead.

        Parameters
        ----------
        key : jax.Array
            This group's PRNG key for this step, already split from the carry.
        step : jax.Array
            The carried step counter, for table-driven overrides.
        group_index : int
            Which hidden group is being drawn for (static).
        shape : tuple of int
            ``(*varshape, num_state)``.
        dtype : dtype-like
            The factor's dtype.

        Returns
        -------
        jax.Array
            The projection vector, shape *shape*.
        """
        bits = brainstate.random.RandomState(key).bernoulli(0.5, shape)
        return jnp.where(bits, 1.0, -1.0).astype(dtype)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #

    def _etp_paths_of_group(self, group: HiddenGroup) -> List[Path]:
        """The ``ParamState`` paths reached by ETP relations of *group*."""
        paths: List[Path] = []
        relation: HiddenParamOpRelation
        for relation in self.graph.hidden_param_op_relations:
            if group not in relation.hidden_groups:
                continue
            for path in relation.trainable_paths.values():
                if path not in paths:
                    paths.append(path)
        return paths

    def _group_dtype(self, group: HiddenGroup) -> Any:
        """The dtype of a group's concatenated hidden slab.

        Read off the model's own hidden states rather than assumed, because
        ``s_tilde`` shares a scan carry with values derived from them.

        Parameters
        ----------
        group : HiddenGroup
            The compiled hidden group.

        Returns
        -------
        numpy.dtype
            The promoted dtype across the group's hidden paths.
        """
        dtypes = [jax.dtypes.result_type(u.get_mantissa(self.hidden_states[p].value))
                  for p in group.hidden_paths]
        return jax.dtypes.result_type(*dtypes) if dtypes else jnp.float32

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        """Allocate the rank-1 factors, the projection key and the step counter."""
        weight_vals = {path: st.value for path, st in self.param_states.items()}

        self.etrace_s = {}
        self.etrace_theta = {}
        for group in self.graph.hidden_groups:
            shape = (*group.varshape, group.num_state)
            # The group's own dtype, not a hard-coded `float32`. `s_tilde` rides
            # in a scan carry alongside `D @ s_tilde`, whose dtype comes from the
            # model; a mismatch is not a silent downcast but a hard
            # "carry input and carry output must have equal types" from
            # `jax.lax.scan`. A float64 model under `jax_enable_x64`, or a
            # float16 one, would fail on the first `MultiStepData` call.
            self.etrace_s[group.index] = brainstate.ShortTermState(
                jnp.zeros(shape, dtype=self._group_dtype(group)))
            for path in self._etp_paths_of_group(group):
                self.etrace_theta[(group.index, path)] = brainstate.ShortTermState(
                    jax.tree.map(u.math.zeros_like, weight_vals[path])
                )
        self.projection_rng = brainstate.ShortTermState(
            self._initial_projection_key())
        self.projection_step = brainstate.ShortTermState(jnp.zeros((), jnp.int32))

        # Last: the base allocates the axis-side state that is not ours.
        super().init_etrace_state(*args, **kwargs)

    def reset_state(self, batch_size: int | None = None, **kwargs: Any) -> None:
        """Reset the factors, the step counter and the projection stream.

        The stream is re-derived from ``projection_key`` rather than advanced, so
        a reset run reproduces the previous one bit-for-bit.
        """
        self.running_index.value = 0
        for st in self.etrace_s.values():
            st.value = jnp.zeros_like(st.value)
        for st in self.etrace_theta.values():
            st.value = jax.tree.map(u.math.zeros_like, st.value)
        self.projection_rng.value = self._initial_projection_key()
        self.projection_step.value = jnp.zeros((), jnp.int32)

    def _get_etrace_data(self) -> Dict[str, Any]:
        """The carry: both factor families, the key and the step counter."""
        return {
            _S_TILDE: {k: v.value for k, v in self.etrace_s.items()},
            _THETA_TILDE: {k: v.value for k, v in self.etrace_theta.items()},
            _KEY: self.projection_rng.value,
            _STEP: self.projection_step.value,
        }

    def _assign_etrace_data(self, etrace_vals: Dict[str, Any]) -> None:
        for k, val in etrace_vals[_S_TILDE].items():
            self.etrace_s[k].value = val
        for k, val in etrace_vals[_THETA_TILDE].items():
            self.etrace_theta[k].value = val
        self.projection_rng.value = etrace_vals[_KEY]
        self.projection_step.value = etrace_vals[_STEP]

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Dict:
        """The parameter-side factors associated with *weight*.

        Parameters
        ----------
        weight : brainstate.ParamState or Path
            The weight whose factors are requested.

        Returns
        -------
        dict
            ``{(group index, path): theta_tilde}`` for that weight.

        Raises
        ------
        ValueError
            If the weight has no random-projection factor.
        """
        self._assert_compiled()
        path = (
            weight if not isinstance(weight, brainstate.ParamState)
            else next(p for p, st in self.param_states.items() if st is weight)
        )
        found = {k: st.value for k, st in self.etrace_theta.items() if k[1] == path}
        if not found:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')
        return found

    # ------------------------------------------------------------------ #
    # the projector: nu^T J_f
    # ------------------------------------------------------------------ #

    def _project(
        self,
        group: HiddenGroup,
        nu: jax.Array,
        etrace_xs_at_t: Dict[Any, jax.Array],
        etrace_ys_at_t: Dict[Any, jax.Array],
        weight_vals: Dict[Path, PyTree],
    ) -> Dict[Path, PyTree]:
        r""":math:`\nu^\top J_f`, per ``ParamState`` path.

        No new per-primitive kernel: this is
        ``solve_rule(nu, instant_rule(x, df, weights))``, the same composition the
        framework already uses for the instantaneous part of a gradient, so it
        inherits each primitive's documented regime exactly — including
        ``einsum``'s shared-axis restriction, conv's bias reduction and LoRA's
        chaining through the current weights.
        """
        temp: Dict[Path, PyTree] = {}
        folded_paths: set = set()
        batched_paths: set = set()

        relation: HiddenParamOpRelation
        for relation in self.graph.hidden_param_op_relations:
            if group not in relation.hidden_groups:
                continue
            if is_batched_primitive(relation.primitive):
                batched_paths.update(relation.trainable_paths.values())

            weights_dict = relation_weights_dict(relation, weight_vals)
            x = (
                None if relation.primitive is etp_elemwise_p
                else etrace_xs_at_t[id(relation.x_var)]
            )
            df = etrace_ys_at_t[etrace_df_key(relation.y, group.index)]
            instant = relation_instant_term(
                relation, group, x, df, weights_dict,
                fast_solve=self.fast_solve, trace_dtype=None,
            )
            proj_dict, batch_folded = relation_solve_to_param(
                relation, group, nu, instant, weight_vals,
                fast_solve=self.fast_solve,
            )
            if batch_folded:
                folded_paths.update(relation.trainable_paths.values())
            _route_grads_by_path(relation, proj_dict, weight_vals, temp)

        reduce_param_batch_axes(temp, weight_vals, folded_paths, batched_paths)
        return temp

    # ------------------------------------------------------------------ #
    # the step
    # ------------------------------------------------------------------ #

    def _make_scan_fn(self, weight_vals: Dict[Path, PyTree]):
        """Build the per-step rank-1 update (the scan body)."""

        def scan_fn(carry: Dict[str, Any], jacobians: Any) -> Tuple[Dict[str, Any], None]:
            etrace_xs_at_t, etrace_ys_at_t, hid_group_jacobians = jacobians
            eps = self.projection_eps

            step = carry[_STEP]
            groups = self.graph.hidden_groups
            # One split per step, one subkey per group: the groups' draws must be
            # independent, and the carried key must advance whether or not any
            # particular group drew.
            rng = brainstate.random.RandomState(carry[_KEY])
            subkeys = jnp.asarray(rng.split_key(len(groups)))
            new_key = rng.value

            new_s: Dict[int, jax.Array] = dict(carry[_S_TILDE])
            new_theta: Dict[Any, PyTree] = dict(carry[_THETA_TILDE])

            group: HiddenGroup
            for group in groups:
                gi = group.index
                s_tilde = carry[_S_TILDE][gi]
                nu = self._draw_projection(
                    subkeys[gi], step, gi, s_tilde.shape, s_tilde.dtype)

                d_s = _apply_full_transition(hid_group_jacobians[gi], s_tilde)
                d_s = u.get_mantissa(d_s)
                proj = self._project(
                    group, nu, etrace_xs_at_t, etrace_ys_at_t, weight_vals)

                paths = self._etp_paths_of_group(group)
                theta_g = {p: carry[_THETA_TILDE][(gi, p)] for p in paths}

                # rho0 may be any positive draw-independent scalar and rho1 any
                # positive *even* function of the draw without touching
                # unbiasedness (the cross terms are odd, and rho1 is even because
                # both its norms are). The choice below is the variance-balancing
                # one from the UORO paper; the eps guard makes the first step,
                # where every norm is zero, finite instead of NaN.
                rho0 = jnp.sqrt((_tree_norm(theta_g) + eps) / (_tree_norm(d_s) + eps))
                rho1 = jnp.sqrt(
                    (_tree_norm(proj) + eps) / (_tree_norm(nu) + eps))

                # Narrowed back to the carry's own dtype: the normalisers are
                # float32-or-wider by design, and the carry's dtype is part of
                # the scan's contract, not a preference.
                new_s[gi] = _at_precision_of(rho0 * d_s + rho1 * nu, s_tilde)
                for p in paths:
                    new_theta[(gi, p)] = jax.tree.map(
                        lambda th, pr: _at_precision_of(th / rho0 + pr / rho1, th),
                        theta_g[p], proj[p], is_leaf=u.math.is_quantity,
                    )

            return (
                {
                    _S_TILDE: new_s,
                    _THETA_TILDE: new_theta,
                    _KEY: new_key,
                    _STEP: step + 1,
                },
                None,
            )

        return scan_fn

    def _make_etrace_stepper(self, weight_vals: Dict[Path, PyTree]):
        """Always the fused per-step stepper.

        Never ``None``: the stack-then-scan path would hold the full
        ``(P S) x (P S)`` transition Jacobian for every step of the window at
        once, and keeping that transient per-step is the whole reason the memory
        table is affordable.
        """
        return self._make_scan_fn(weight_vals)

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Dict[str, Any],
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: HiddenGroupJacobian,
        weight_vals: Dict[Path, PyTree],
        input_is_multi_step: bool,
    ) -> Dict[str, Any]:
        """Roll the rank-1 factors over one step or one window."""
        jacobians = (
            hid2weight_jac_single_or_multi_times[0],
            hid2weight_jac_single_or_multi_times[1],
            hid2hid_jac_single_or_multi_times,
        )
        scan_fn = self._make_scan_fn(weight_vals)
        if input_is_multi_step:
            return jax.lax.scan(scan_fn, hist_etrace_vals, jacobians)[0]
        return scan_fn(hist_etrace_vals, jacobians)[0]

    # ------------------------------------------------------------------ #
    # the boundary contraction
    # ------------------------------------------------------------------ #

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[str, Any],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ) -> Any:
        r"""``sum_g (signal_g . s_tilde_g) * theta_tilde_g``, plus the direct terms.

        A scalar per group times a parameter-shaped array — the whole reason the
        parameter factor carries no hidden axis.
        """
        dG_weights: Dict[Path, Any] = {path: None for path in self.param_states}

        for group in self.graph.hidden_groups:
            gi = group.index
            signal = u.get_mantissa(dl_to_hidden_groups[gi])
            s_tilde = etrace_h2w_at_t[_S_TILDE][gi]
            scale = jnp.sum(signal * s_tilde)
            for path in self._etp_paths_of_group(group):
                theta = etrace_h2w_at_t[_THETA_TILDE][(gi, path)]
                _update_dict(dG_weights, path, _scale_tree(theta, scale))

        # the non-etrace weight gradients (reverse-AD, in-window)
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # the in-window direct term for etrace parameters (multi-step only)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights
