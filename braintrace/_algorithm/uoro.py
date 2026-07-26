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

"""UORO — Unbiased Online Recurrent Optimization (Tallec & Ollivier, 2018).

Every other trace in this repository is *biased*: it drops something (the
recurrent mixing, the cross-position influence, the terms outside a SnAp
neighbourhood) and the gradient it produces is systematically off, however long
you train. UORO drops nothing and is wrong *at random* instead: it replaces the
influence matrix with a rank-1 outer product whose two factors are re-randomised
every step so that the estimate is right **in expectation**.

The trade is variance for bias, at a carrier size comparable to the anchored
per-parameter trace. See :class:`UORO` for what "unbiased" is scoped to --
it is narrower than the word suggests -- and
:class:`~braintrace._algorithm.random_projection_vjp.RandomProjectionVjpAlgorithm`
for the update and the proof sketch.
"""

from __future__ import annotations

from typing import Any, Optional

import brainstate
import jax

from braintrace._compiler import ControlFlowPolicy, DEFAULT_MAX_JACOBIAN_ELEMENTS
from .axes import ETraceConfig
from .random_projection_vjp import RandomProjectionVjpAlgorithm

__all__ = ['UORO']


class UORO(RandomProjectionVjpAlgorithm):
    r"""Unbiased Online Recurrent Optimization.

    The coordinate is
    ``ETraceConfig(trace_factorization='random_projection', recurrence_scope='coupled')``.
    Both halves are load-bearing:

    * **random_projection** replaces the per-parameter influence trace with one
      rank-1 pair per hidden group, ``eps_tilde[j, u] ~= s_tilde[u] *
      theta_tilde[j]``, re-randomised each step with a Rademacher draw.
    * **coupled** is required, not merely recommended (matrix rule 11). A rank-1
      *unbiased* estimator of an already-*biased* recursion would be strictly
      worse than the biased recursion itself: same asymptotic error, more
      variance, no memory saved — the anchored per-parameter trace is already the
      smaller carrier. ``coupled`` is the cheapest scope whose transition
      actually contains the hidden-to-hidden ETP mixing, so it is the coordinate
      where the projection buys something. Measured: rolling the block-diagonal
      transition converges cleanly onto the *biased* trace and never onto the
      exact one.

    What follows is the claim in full, because "unbiased" on its own overstates
    it: **UORO is an unbiased estimator of the exact within-group influence
    recursion that the compiled transition defines** — that is, the saturating end
    of the :class:`~braintrace.SnAp` scale, at SnAp-1 memory, in expectation. It
    does not repair cross-group coupling, the F-31 instantaneous tail, or any
    primitive's own solve regime. On a single hidden group with a
    position-preserving elementwise tail, that recursion is itself exact, so there
    UORO is unbiased for BPTT; elsewhere it is unbiased for the block-local
    recursion. It runs on every model either way.

    Parameters
    ----------
    model : brainstate.nn.Module
        The one-step model.
    name : str, optional
        Node name.
    vjp_method : str, optional
        ``'multi-step'`` (default) or ``'single-step'``. The finite-window oracle
        path — and hence every acceptance criterion — uses ``'multi-step'``.
    fast_solve : bool, optional
        Whether registered closed-form kernels may be used when building the
        projector. Default ``True``.
    control_flow : ControlFlowPolicy, optional
        Control-flow canonicalization policy.
    projection_key : int or jax.Array, optional
        Seed for the projection stream. Default ``42``. Two different keys give
        two different gradients; ``reset_state`` re-derives the stream from this
        value, so a reset run repeats bit-for-bit.
    projection_eps : float, optional
        Guard added to every norm before its ratio. Default ``1e-12``; ``0.0``
        makes the first step NaN, since both norms start at zero.
    random_feedback_key : jax.Array, optional
        Required when combining with ``learning_signal='random_feedback'``.
    snap_max_jacobian_elements : int, optional
        Passed through; unused at ``recurrence_scope='coupled'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate, braintrace, jax.numpy as jnp
        >>> class Net(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.w = brainstate.ParamState(0.1 * jnp.ones((4, 4)))
        ...         self.h = brainstate.HiddenState(jnp.ones((1, 4)) * 0.4)
        ...     def update(self, x):
        ...         self.h.value = jnp.tanh(x + braintrace.matmul(self.h.value, self.w.value))
        ...         return self.h.value
        >>> model = Net()
        >>> brainstate.nn.init_all_states(model, batch_size=1)
        >>> learner = braintrace.UORO(model, projection_key=0)
        >>> learner.compile_graph(braintrace.MultiStepData(jnp.zeros((1, 1, 4))))
        >>> learner.init_etrace_state()

    Notes
    -----
    **Variance grows with the number of window boundaries.** The estimate is
    unbiased, not low-variance: a single run can be far from the mean, and
    averaging is the caller's job. Antithetic sampling does *not* help here and
    looks like it should — flipping the sign of the entire draw sequence leaves
    the estimate **bit-identical**, because ``rho1`` is even and both factors flip
    together, so their product is invariant.

    See Also
    --------
    braintrace.SnAp : the deterministic scale whose saturating end this
        estimates, at higher memory.
    braintrace.D_RTRL : the diagonal-scope biased trace.
    """

    __module__ = 'braintrace'

    #: UORO *is* this coordinate; ``coupled`` is not a default to be overridden.
    _default_config = ETraceConfig(
        trace_factorization='random_projection', recurrence_scope='coupled')

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'multi-step',
        fast_solve: bool = True,
        control_flow: Optional[ControlFlowPolicy] = None,
        projection_key: Any = 42,
        projection_eps: float = 1e-12,
        random_feedback_key: Optional[jax.Array] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ) -> None:
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            control_flow=control_flow,
            config=self._default_config,
            projection_key=projection_key,
            projection_eps=projection_eps,
            random_feedback_key=random_feedback_key,
            snap_max_jacobian_elements=snap_max_jacobian_elements,
        )
