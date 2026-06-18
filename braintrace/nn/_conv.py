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

from braintrace._etrace_op import conv as etp_conv

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


def _etp_conv_op(self, x, params):
    """Route a convolution through the ETP ``conv`` primitive.

    Shared ``_conv_op`` override installed on :class:`Conv1d`, :class:`Conv2d`
    and :class:`Conv3d`. Using :func:`braintrace.conv` instead of a plain JAX
    convolution is what makes the kernel eligible for online-learning trace
    computation; all convolution hyper-parameters are taken from ``self``.

    Parameters
    ----------
    x : ArrayLike
        Input feature map.
    params : dict
        Parameter dict holding the convolution ``'weight'`` and an optional
        ``'bias'``.

    Returns
    -------
    ArrayLike
        The convolution output.
    """
    w = params['weight']
    if self.w_mask is not None:
        w = w * self.w_mask
    b = params.get('bias')
    return etp_conv(
        x, w, b,
        strides=self.stride,
        padding=self.padding,
        lhs_dilation=self.lhs_dilation,
        rhs_dilation=self.rhs_dilation,
        feature_group_count=self.groups,
        dimension_numbers=self.dimension_numbers,
    )


class Conv1d(brainstate.nn.Conv1d):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Conv1d.__doc__ or '').replace('brainstate', 'braintrace')
    _conv_op = _etp_conv_op


class Conv2d(brainstate.nn.Conv2d):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Conv2d.__doc__ or '').replace('brainstate', 'braintrace')
    _conv_op = _etp_conv_op


class Conv3d(brainstate.nn.Conv3d):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Conv3d.__doc__ or '').replace('brainstate', 'braintrace')
    _conv_op = _etp_conv_op
