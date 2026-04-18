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

# -*- coding: utf-8 -*-

import brainstate
from brainstate.nn._normalizations import NormalizationParamState

__all__ = [
    'BatchNorm0d',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'RMSNorm',
    'GroupNorm',
]

_NORM_PARAM = NormalizationParamState


class BatchNorm0d(brainstate.nn.BatchNorm0d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.BatchNorm0d.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class BatchNorm1d(brainstate.nn.BatchNorm1d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.BatchNorm1d.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class BatchNorm2d(brainstate.nn.BatchNorm2d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.BatchNorm2d.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class BatchNorm3d(brainstate.nn.BatchNorm3d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.BatchNorm3d.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class LayerNorm(brainstate.nn.LayerNorm):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.LayerNorm.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class RMSNorm(brainstate.nn.RMSNorm):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.RMSNorm.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


class GroupNorm(brainstate.nn.GroupNorm):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.GroupNorm.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)
