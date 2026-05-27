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

"""Neural-network layers wired for Eligibility Trace Propagation (ETP).

This subpackage mirrors a subset of :mod:`brainstate.nn` but routes the
trainable forward passes through ETP primitives (``braintrace.matmul``,
``braintrace.conv``, ``braintrace.sparse_matmul``, ``braintrace.lora_matmul``,
``braintrace.element_wise``). As a result, the parameters of these layers are
automatically recognised by the ETP compiler and become eligible for online
learning, while the public construction and call signatures stay compatible
with their :mod:`brainstate.nn` counterparts.

The exported building blocks fall into four groups:

- **Linear maps** — :class:`Linear`, :class:`SignedWLinear`,
  :class:`SparseLinear`, :class:`LoRA`.
- **Convolutions** — :class:`Conv1d`, :class:`Conv2d`, :class:`Conv3d`.
- **Read-outs** — :class:`LeakyRateReadout`.
- **Recurrent cells** — :class:`ValinaRNNCell`, :class:`GRUCell`,
  :class:`MGUCell`, :class:`LSTMCell`, :class:`URLSTMCell`,
  :class:`MinimalRNNCell`, :class:`MiniGRU`, :class:`MiniLSTM`,
  :class:`LRUCell`.

Activation, normalisation and pooling layers are intentionally not
re-implemented here; accessing them through ``braintrace.nn`` emits a
:class:`DeprecationWarning` and forwards to :mod:`brainstate.nn` /
:mod:`brainstate.state`.
"""

from ._conv import Conv1d, Conv2d, Conv3d
from ._linear import Linear, SignedWLinear, SparseLinear, LoRA
from ._readout import LeakyRateReadout
from ._rnn import ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell, MinimalRNNCell, MiniGRU, MiniLSTM, LRUCell

__all__ = [
    # conv
    'Conv1d', 'Conv2d', 'Conv3d',
    # linear
    'Linear', 'SignedWLinear', 'SparseLinear', 'LoRA',
    # readout
    'LeakyRateReadout',
    # rnn
    'ValinaRNNCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'URLSTMCell',
    'MinimalRNNCell', 'MiniGRU', 'MiniLSTM', 'LRUCell',
]


def __getattr__(name):
    import warnings
    if name in ['IF', 'LIF', 'ALIF', 'Expon', 'Alpha', 'DualExpon', 'STP', 'STD']:
        warnings.warn(
            f'braintrace.nn.{name} is deprecated. Use brainstate.state.{name} instead.',
            DeprecationWarning,
            stacklevel=2
        )
        import brainstate.state
        return getattr(brainstate.state, name)

    if name in [
        'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
        'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
        'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU',
        'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax',
        'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
        'Identity', 'SpikeBitwise',

        'Flatten', 'Unflatten',
        'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
        'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
        'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
        'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',

        'BatchNorm0d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
        'LayerNorm', 'RMSNorm', 'GroupNorm',
    ]:
        warnings.warn(
            f'braintrace.nn.{name} is deprecated. Use brainstate.nn.{name} instead.',
            DeprecationWarning,
            stacklevel=2
        )
        import brainstate
        return getattr(brainstate.nn, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
