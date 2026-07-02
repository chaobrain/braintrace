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

"""Tests for the general linear-contraction ETP primitive and
:func:`einsum` API."""

from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

import braintrace
from braintrace._op.einsum import EinsumSpec, parse_etp_einsum

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


class TestParser:

    def test_dense_equation(self):
        s = parse_etp_einsum('bk,kn->bn')
        assert s == EinsumSpec('bk', 'kn', 'bn', 'b', diagonal='n',
                               contracted='k', shared='')

    def test_grouped_equation(self):
        s = parse_etp_einsum('bgk,gkn->bgn')
        assert s.batch == 'b'
        assert s.diagonal == 'gn'
        assert s.contracted == 'k'
        assert s.shared == ''

    def test_per_head_equation(self):
        s = parse_etp_einsum('bhd,hde->bhe')
        assert s.diagonal == 'he'
        assert s.contracted == 'd'
        assert s.shared == ''

    def test_shared_axis_equation_classified(self):
        s = parse_etp_einsum('btk,kn->btn')
        assert s.diagonal == 'n'
        assert s.contracted == 'k'
        assert s.shared == 't'

    def test_spaces_normalized(self):
        assert parse_etp_einsum(' bk , kn -> bn ') == parse_etp_einsum('bk,kn->bn')

    @pytest.mark.parametrize('bad', [
        'bk,kn',            # no explicit output
        'bk->b',            # one operand
        'bk,kn,nm->bm',     # three operands
        'bk,kn->bnz',       # output letter from nowhere
        'Bk,kn->Bn',        # uppercase
        'b...k,kn->b...n',  # ellipsis
        'bkk,kn->bn',       # repeated letter within a spec
        'bk,bn->bn',        # batch letter inside weight spec
        'kb,kn->bn',        # x does not lead with the batch letter
        'bk,kv->bn',        # weight letter v in neither x nor output
    ])
    def test_rejections(self, bad):
        with pytest.raises(ValueError):
            parse_etp_einsum(bad)
