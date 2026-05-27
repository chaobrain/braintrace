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

import pytest

import braintrace
import braintrace._legacy as legacy

_LEGACY_NAMES = [
    'ETraceOp', 'MatMulOp', 'ElemWiseOp', 'ConvOp', 'SpMatMulOp', 'LoraOp',
    'ETraceParam', 'ElemWiseParam', 'NonTempParam',
    'FakeETraceParam', 'FakeElemWiseParam',
]


@pytest.mark.parametrize('name', _LEGACY_NAMES)
def test_legacy_access_warns_and_returns_class(name):
    with pytest.warns(DeprecationWarning):
        obj = getattr(braintrace, name)
    assert obj is getattr(legacy, name)


@pytest.mark.parametrize('name', _LEGACY_NAMES)
def test_legacy_names_not_in_all(name):
    assert name not in braintrace.__all__


def test_from_import_warns():
    with pytest.warns(DeprecationWarning):
        from braintrace import MatMulOp  # noqa: F401


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = braintrace.ThisNameDoesNotExist


def test_legacy_names_in_dir():
    d = dir(braintrace)
    for name in _LEGACY_NAMES:
        assert name in d
