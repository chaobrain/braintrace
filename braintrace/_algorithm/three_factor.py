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

"""Three-factor learning: eligibility trace x a user-supplied modulator.

The two factors of an eligibility trace -- presynaptic activity and postsynaptic
sensitivity -- are local. The third factor is not: it is a global, delayed,
low-dimensional signal (dopamine, reward prediction error, a task cue) that gates
whether the accumulated eligibility becomes a weight change at all.

Every other rule in this repository takes that third factor from reverse-AD:
``symmetric`` uses ``dL/dh`` exactly, ``random_feedback`` uses a frozen
projection of it. :class:`ThreeFactor` takes it from the caller instead, which is
what makes reward-modulated and neuromodulatory rules expressible without a new
engine.
"""

from __future__ import annotations

from typing import Any, Optional

import brainstate

from braintrace._compiler import ControlFlowPolicy, DEFAULT_MAX_JACOBIAN_ELEMENTS
from .axes import ETraceConfig
from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = ['ThreeFactor']


class ThreeFactor(ParamDimVjpAlgorithm):
    r"""Reward-modulated eligibility-trace learning.

    The coordinate is :class:`~braintrace.D_RTRL`'s with
    ``learning_signal='modulatory'``: a per-parameter trace, diagonal recurrence
    scope, and a learning signal supplied by the caller rather than by reverse-AD.

    .. math::

        \frac{\partial L}{\partial \theta} \;\leftarrow\;
        \sum_g \mathrm{expand}(m) \cdot \varepsilon_g

    **Replace, not multiply.** The modulator *is* the learning signal; it does not
    scale ``dL/dh``. Multiplying would give a four-factor rule, and would make the
    degenerate check -- set the modulator to ``dL/dh`` and recover ``symmetric``
    exactly -- impossible to satisfy, leaving the axis with no coordinate at which
    it reduces to the rule it generalises.

    **One array, expanded to every group.** A scalar reward is valid for any model
    whatever its HiddenGroup count. There is deliberately no per-group sequence
    spelling: binding the signal to the hidden-group decomposition, whose size is
    a property of the compiled graph rather than of the task, is what made OSTTP
    non-general. Expansion follows
    :func:`~braintrace._algorithm.vjp_base.expand_modulator_to_group` -- shape
    driven, never group indexed.

    **Single-step only, and it raises otherwise.** Under multi-step,
    ``_solve_weight_gradients`` adds the within-window reverse-AD gradient of the
    ETP parameters on top of the trace contraction, so replacing the *boundary*
    signal would leave that in-window half unmodulated: a hybrid that is part
    three-factor rule and part plain loss gradient. Single-step routes every ETP
    contribution through the replaced signal, and makes the modulator per step,
    which is what a neuromodulator is. A consequence worth stating: because the
    window is one step, ``update_schedule`` has nothing to schedule and stays out
    of this preset.

    Parameters
    ----------
    model : brainstate.nn.Module
        The one-step model.
    name : str, optional
        Node name.
    vjp_method : str, optional
        Must be ``'single-step'`` (the default); anything else raises.
    modulator : array_like or Quantity, optional
        The initial standing modulator, equivalent to assigning
        :attr:`~braintrace._algorithm.vjp_base.ETraceVjpAlgorithm.modulator`
        after construction. A scalar, an array shaped like a group's
        ``varshape``, or an array broadcastable to ``(*varshape, num_state)``.
        Leaving it ``None`` is fine as long as one is supplied before the first
        ``update()``; there is no fallback to ``symmetric``.
    fast_solve : bool, optional
        Whether registered closed-form kernels may be used. Default ``True``.
    trace_dtype : DTypeLike, optional
        Reduced trace precision, as in :class:`~braintrace.D_RTRL`.
    chunked_trace : bool, optional
        Whether to roll the trace in chunks. Default ``True``.
    control_flow : ControlFlowPolicy, optional
        Control-flow canonicalization policy.
    snap_max_jacobian_elements : int, optional
        Passed through; unused at ``recurrence_scope='diagonal'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate, braintrace, jax.numpy as jnp
        >>> class Net(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.w = brainstate.ParamState(0.1 * jnp.ones((4, 4)))
        ...         self.h = brainstate.HiddenState(jnp.zeros((1, 4)))
        ...     def update(self, x):
        ...         self.h.value = jnp.tanh(x + braintrace.matmul(self.h.value, self.w.value))
        ...         return self.h.value
        >>> model = Net()
        >>> brainstate.nn.init_all_states(model, batch_size=1)
        >>> learner = braintrace.ThreeFactor(model)
        >>> learner.compile_graph(jnp.zeros((1, 4)))
        >>> learner.init_etrace_state()

    Then drive it with a reward per step, either per call or as a standing value:

    .. code-block:: python

        >>> out = learner.update(jnp.zeros((1, 4)), modulator=0.5)
        >>> learner.modulator = -1.0        # standing, until reassigned

    The keyword takes precedence for the call it appears on.

    Notes
    -----
    Under single-step, every **plain** (non-ETP) parameter's gradient is exactly
    zero (F-33) -- not merely truncated. Since this preset is single-step by
    construction, a model whose parameters are partly plain will train only its
    ETP-routed parameters here. Route the parameters you intend to modulate
    through an ETP primitive (``braintrace.matmul`` and friends).

    The modulator is a genuine data dependency of the traced computation, not a
    lazily-read attribute: it is read synchronously at the top of ``update()``
    (see ``_get_update_aux``), because an outer transform may stage the forward
    trace and invoke the backward rule only after ``update()`` has returned.

    See Also
    --------
    braintrace.D_RTRL : the same coordinate with ``learning_signal='symmetric'``.
    braintrace.ETraceConfig : the full axis space.
    """

    __module__ = 'braintrace'

    #: D_RTRL's coordinate with the learning signal replaced.
    _default_config = ETraceConfig(
        trace_factorization='per_param',
        temporal_recursion='jacobian',
        recurrence_scope='diagonal',
        learning_signal='modulatory',
    )

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        modulator: Any = None,
        fast_solve: bool = True,
        trace_dtype: Any = None,
        chunked_trace: bool = True,
        control_flow: Optional[ControlFlowPolicy] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ) -> None:
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            trace_dtype=trace_dtype,
            chunked_trace=chunked_trace,
            control_flow=control_flow,
            config=self._default_config,
            snap_max_jacobian_elements=snap_max_jacobian_elements,
        )
        self.modulator = modulator
