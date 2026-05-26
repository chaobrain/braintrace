# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from .io_dim_vjp import IODimVjpAlgorithm

__all__ = [
    'pp_prop',  # the diagonally approximated algorithm with the input-output dimension complexity
    'ES_D_RTRL',
]


class pp_prop(IODimVjpAlgorithm):
    r"""
    The online gradient computation algorithm with the diagonal approximation
    and the input-output dimensional complexity.

    ``pp_prop`` is the canonical name for the input-output-dimension eligibility
    trace algorithm implemented by :py:class:`IODimVjpAlgorithm`. It computes the
    gradients of the weights with the diagonal approximation and the
    input-output dimensional complexity, following the learning rule:

    $$
    \begin{aligned}
    & \boldsymbol{\epsilon}^t \approx \boldsymbol{\epsilon}_{\mathbf{f}}^t \otimes \boldsymbol{\epsilon}_{\mathbf{x}}^t \\
    & \boldsymbol{\epsilon}_{\mathbf{x}}^t=\alpha \boldsymbol{\epsilon}_{\mathbf{x}}^{t-1}+\mathbf{x}^t \\
    & \boldsymbol{\epsilon}_{\mathbf{f}}^t=\alpha \operatorname{diag}\left(\mathbf{D}^t\right) \circ \boldsymbol{\epsilon}_{\mathbf{f}}^{t-1}+(1-\alpha) \operatorname{diag}\left(\mathbf{D}_f^t\right) \\
    & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
    \end{aligned}
    $$

    For more details, please see `the ES-D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    This subclass inherits all behavior from :py:class:`IODimVjpAlgorithm` without
    modification; it exists to provide the canonical ``pp_prop`` name. See
    :py:class:`IODimVjpAlgorithm` for the full parameter list and a usage example.

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

    __module__ = 'braintrace'


ES_D_RTRL = pp_prop
