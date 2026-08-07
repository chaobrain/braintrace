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

from __future__ import annotations

from typing import Any

import brainstate

from braintrace._op import conv as etp_conv
from braintrace._typing import ArrayLike

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


def _adapt_doc(doc: str | None) -> str:
    """Retarget an upstream ``brainstate`` docstring at ``braintrace``.

    Besides the name substitution, this closes bullet lists that the upstream
    text leaves open. Several ``brainstate.nn`` convolution docstrings follow a
    bullet item directly with a same- or lesser-indented paragraph (``Default:
    1.``) or with the next NumPy-doc parameter, which docutils reports as
    ``Bullet list ends without a blank line; unexpected unindent`` under the
    ``nitpicky`` docs build. A blank line is inserted after the final item of
    each such list. Doing it structurally rather than by matching the exact
    upstream wording keeps the fix working when that wording changes.

    Parameters
    ----------
    doc : str or None
        The upstream docstring, or ``None`` when the class carries none.

    Returns
    -------
    str
        The adapted docstring; the empty string when ``doc`` is ``None``.
    """
    lines = (doc or '').replace('brainstate', 'braintrace').split('\n')
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else ''
        prev_is_item = prev.lstrip().startswith('- ')
        if prev_is_item and line.strip():
            prev_indent = len(prev) - len(prev.lstrip())
            indent = len(line) - len(line.lstrip())
            # A deeper indent continues the item; another ``- `` continues the
            # list. Anything else closes it and needs a separating blank line.
            if indent <= prev_indent and not line.lstrip().startswith('- '):
                out.append('')
        out.append(line)
    return '\n'.join(out)


def _etp_conv_op(self: Any, x: ArrayLike, params: dict[str, Any]) -> ArrayLike:
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
    __doc__ = _adapt_doc(brainstate.nn.Conv1d.__doc__)
    _conv_op = _etp_conv_op


class Conv2d(brainstate.nn.Conv2d):
    __module__ = 'braintrace.nn'
    __doc__ = _adapt_doc(brainstate.nn.Conv2d.__doc__)
    _conv_op = _etp_conv_op


class Conv3d(brainstate.nn.Conv3d):
    __module__ = 'braintrace.nn'
    __doc__ = _adapt_doc(brainstate.nn.Conv3d.__doc__)
    _conv_op = _etp_conv_op
