# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = [
    'D_RTRL',
]


class D_RTRL(ParamDimVjpAlgorithm):
    r"""Compute online gradients with the Diagonal RTRL preset.

    ``D_RTRL`` is the canonical name for the parameter-dimension eligibility
    trace algorithm implemented by :class:`ParamDimVjpAlgorithm`. It computes
    the gradients of the weights with the diagonal approximation and the
    parameter dimension complexity, following the learning rule:

    .. math::

        \begin{aligned}
        \boldsymbol{\epsilon}^t
        &\approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}
        + \operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t, \\
        \nabla_{\boldsymbol{\theta}} \mathcal{L}
        &= \sum_{t^{\prime} \in \mathcal{T}}
        \frac{\partial \mathcal{L}^{t^{\prime}}}
        {\partial \mathbf{h}^{t^{\prime}}}
        \circ \boldsymbol{\epsilon}^{t^{\prime}}.
        \end{aligned}

    This formulation follows the D-RTRL estimator presented by Wang et al.
    [1]_ and the original RTRL construction of Williams and Zipser [2]_.

    Parameters
    ----------
    model : brainstate.nn.Module
        Recurrent model whose ETP-routed parameters are trained online.
    name : str, optional
        Name of the algorithm instance.
    vjp_method : {'single-step', 'multi-step'}, optional
        VJP window used to compute the learning signal.
    fast_solve : bool, optional
        Whether to use closed-form per-primitive contractions when available.
    trace_dtype : dtype, optional
        Storage dtype for supported fast-path eligibility traces.
    chunked_trace : bool, optional
        Whether multi-step inputs use the closed-form chunked trace roll.
    control_flow : ControlFlowPolicy, optional
        Control-flow canonicalization policy used during graph compilation.
    config : ETraceConfig, optional
        Learning-rule coordinates. ``None`` uses the D-RTRL preset.
    random_feedback_key : jax.Array, optional
        Key for fixed random-feedback projections requested by ``config``.
    snap_max_jacobian_elements : int, optional
        Maximum permitted size of each SnAp widened block Jacobian.

    See Also
    --------
    ParamDimVjpAlgorithm
        Parameter-dimensional engine implementing this preset.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> model = braintrace.nn.ValinaRNNCell(2, 4, activation='tanh')
        >>> x0 = brainstate.random.randn(2)
        >>> learner = braintrace.compile(model, braintrace.D_RTRL, x0)
        >>> y = learner.update(x0)

    References
    ----------
    .. [1] Wang, C., Dong, X., Ji, Z., Xiao, M., Jiang, J., Liu, X., Huan, Y., &
       Wu, S. (2026). "Model-agnostic linear-memory online learning in spiking
       neural networks." *Nature Communications*.
       https://doi.org/10.1038/s41467-026-68453-w
       (preprint: bioRxiv 2024.09.24.614728)
    .. [2] Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for
       Continually Running Fully Recurrent Neural Networks" (RTRL). *Neural
       Computation*, 1(2), 270-280. https://doi.org/10.1162/neco.1989.1.2.270
    """
