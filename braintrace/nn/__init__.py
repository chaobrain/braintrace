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


from ._conv import *
from ._conv import __all__ as _conv_all
from ._linear import *
from ._linear import __all__ as _linear_all
from ._normalizations import *
from ._normalizations import __all__ as _normalizations_all
from ._readout import *
from ._readout import __all__ as _readout_all
from ._rnn import *
from ._rnn import __all__ as _rnn_all

__all__ = _conv_all + _linear_all + _normalizations_all + _rnn_all + _readout_all
del _conv_all, _linear_all, _normalizations_all, _rnn_all, _readout_all


def __getattr__(name):
    import warnings
    if name in [
        'IF', 'LIF', 'ALIF',
        'Expon', 'Alpha', 'DualExpon', 'STP', 'STD',
    ]:
        warnings.warn(
            f'braintrace.nn.{name} is deprecated. Use brainstate.state.{name} instead.',
            DeprecationWarning,
            stacklevel=2
        )
        import brainpy
        return getattr(brainpy.state, name)

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
    ]:
        warnings.warn(
            f'braintrace.nn.{name} is deprecated. Use brainstate.nn.{name} instead.',
            DeprecationWarning,
            stacklevel=2
        )
        import brainstate
        return getattr(brainstate.nn, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
