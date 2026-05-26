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

"""OSTTP — Online Spatio-Temporal Learning with Target Projection
(Ortner et al., 2023).

OSTTP combines the OSTL / D-RTRL eligibility trace with a DRTP-style *target
projection*: instead of back-propagating :math:`\\partial \\mathcal{L}/\\partial
h` from the readout, each HiddenGroup receives a learning signal formed by a
fixed random projection of the task target, :math:`y^{*}\\,B_l`. This removes the
weight-transport requirement and the backward pass, so learning is forward-only.

See :class:`OSTTP` for the mathematical formulation, references, and an example.
"""

from typing import Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp

from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = ['OSTTP']


class OSTTP(ParamDimVjpAlgorithm):
    r"""Online Spatio-Temporal Learning with Target Projection.

    OSTTP reuses the OSTL / D-RTRL per-parameter eligibility trace but replaces
    the back-propagated learning signal with a **direct random target
    projection** (DRTP):

    .. math::

        \boldsymbol{\epsilon}^t \approx \mathbf{D}^t\,\boldsymbol{\epsilon}^{t-1}
        + \operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t ,
        \qquad
        L_l^t = y^{*\,t}\, B_l ,
        \qquad
        \nabla_{W}\mathcal{L} = \sum_t L^t \circ \boldsymbol{\epsilon}^t ,

    where :math:`y^{*\,t}` is the task target at time :math:`t`, :math:`B_l \in
    \mathbb{R}^{n_\text{target}\times n_l}` is a fixed random feedback matrix for
    HiddenGroup :math:`l` (frozen via ``stop_gradient``), :math:`\mathbf{D}^t` is
    the hidden-to-hidden Jacobian, :math:`\mathbf{D}_f^t` the state-to-output
    Jacobian, and :math:`\mathbf{x}^t` the presynaptic input.

    **How it works.** The eligibility trace carries the temporal credit exactly
    as in :class:`~braintrace.OSTLRecurrent` ('with-H'), but the spatial credit normally
    obtained by back-propagating :math:`\partial \mathcal{L}/\partial h` is
    replaced by a frozen random projection of the target. Because the projection
    matrices :math:`B_l` are fixed, there is no weight transport and no backward
    pass — the rule is fully forward and update-unlocked in both space and time.

    Parameters
    ----------
    model : brainstate.nn.Module
        The SNN whose weights are trained online.
    B_list : Sequence[jax.Array]
        One feedback matrix per HiddenGroup, each of shape
        ``(n_target, n_l)``. Frozen via ``stop_gradient`` at construction; the
        count and trailing dimension are validated against the compiled graph.
    target_timing : {'per-step', 'sequence-end'}, default 'per-step'
        ``'per-step'`` requires ``y_target`` at every :meth:`update` call.
        ``'sequence-end'`` zeros the learning signal on intermediate steps (the
        trace still accumulates) and applies the projection only when
        ``y_target`` is supplied.
    name : str, optional
        Name of the algorithm instance.
    vjp_method, fast_solve
        Forwarded verbatim to :class:`~braintrace.ParamDimVjpAlgorithm`.

    Raises
    ------
    ValueError
        If ``target_timing`` is invalid; if ``len(B_list)`` differs from the
        number of HiddenGroups; if a matrix's trailing dimension does not match
        its HiddenGroup width; or if ``target_timing='per-step'`` and
        ``y_target`` is omitted from an :meth:`update` call.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import braintrace
        >>>
        >>> class Net(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
        ...         self.out = braintrace.nn.Linear(20, 1)
        ...     def update(self, x):
        ...         return x >> self.cell >> self.out
        >>>
        >>> model = Net()
        >>> _ = brainstate.nn.init_all_states(model)
        >>> # one (n_target, n_l) feedback matrix per HiddenGroup (here n_l = 20)
        >>> B = jax.random.normal(jax.random.PRNGKey(0), (1, 20))
        >>> learner = braintrace.OSTTP(model, B_list=[B])
        >>> x0 = brainstate.random.randn(1)
        >>> learner.compile_graph(x0)
        >>> y = learner.update(x0, y_target=brainstate.random.randn(1))

    References
    ----------
    .. [1] Ortner, T., Pes, L., Gentinetta, J., Frenkel, C., & Pantazi, A.
       (2023). "Online Spatio-Temporal Learning with Target Projection."
       *2023 IEEE 5th International Conference on Artificial Intelligence
       Circuits and Systems (AICAS)*, 1-5.
       https://doi.org/10.1109/AICAS57966.2023.10168623 (arXiv:2304.05124)
    .. [2] Frenkel, C., Lefebvre, M., & Bol, D. (2021). "Learning Without
       Feedback: Fixed Random Learning Signals Allow for Feedforward Training of
       Deep Neural Networks" (DRTP). *Frontiers in Neuroscience*, 15, 629892.
       https://doi.org/10.3389/fnins.2021.629892
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        B_list: Sequence[jax.Array],
        target_timing: str = 'per-step',
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        **kwargs,
    ):
        if target_timing not in ('per-step', 'sequence-end'):
            raise ValueError(
                f"target_timing must be 'per-step' or 'sequence-end'; got {target_timing!r}"
            )
        super().__init__(
            model, name=name, vjp_method=vjp_method, fast_solve=fast_solve, **kwargs
        )
        self._B_list = tuple(jax.lax.stop_gradient(B) for B in B_list)
        self.target_timing = target_timing
        self._current_y_target: Optional[jax.Array] = None

    def compile_graph(self, *args) -> None:
        super().compile_graph(*args)
        n_groups = len(self.graph.hidden_groups)
        if len(self._B_list) != n_groups:
            raise ValueError(
                f'B_list has {len(self._B_list)} entries but model has {n_groups} '
                f'HiddenGroup(s). One B matrix per HiddenGroup is required.'
            )
        for B, group in zip(self._B_list, self.graph.hidden_groups):
            n_l = int(group.varshape[-1])
            if B.shape[1] != n_l:
                raise ValueError(
                    f'B_list[{group.index}].shape[1] == {B.shape[1]} but HiddenGroup '
                    f'{group.index} has n_l={n_l}.'
                )

    def update(self, x, y_target=None):
        """Call ``super().update(x)`` after stashing ``y_target`` for the hook."""
        if self.target_timing == 'per-step' and y_target is None:
            raise ValueError(
                "OSTTP(target_timing='per-step') requires y_target at every update() call."
            )
        self._current_y_target = y_target
        try:
            return super().update(x)
        finally:
            self._current_y_target = None

    def _compute_learning_signal(self, dl_autodiff, args):
        """Replace reverse-AD ``dL/dh`` with ``B_l @ y_target`` per HiddenGroup."""
        y_target = self._current_y_target
        if y_target is None:
            # target_timing='sequence-end' with no y_target: zero out so traces
            # accumulate without emitting a weight update this step.
            return [jnp.zeros_like(s) for s in dl_autodiff]
        out = []
        for gid, s in enumerate(dl_autodiff):
            B = self._B_list[gid]
            projected = y_target @ B  # (batch, n_l)
            # Reshape projected into the autodiff signal shape (which has a
            # trailing num_state axis appended by concat_hidden).
            # s shape == (*varshape, num_state); projected shape == (*varshape,)
            target_shape = s.shape
            expanded = projected.reshape(target_shape[:-1] + (1,))
            # Broadcast across the num_state tail.
            out.append(jnp.broadcast_to(expanded, target_shape).astype(s.dtype))
        return out
