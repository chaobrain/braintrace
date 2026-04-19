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

import unittest


class TestPublicAPI(unittest.TestCase):
    def test_subpackage_exports(self):
        import braintrace._snn_algorithms as pkg
        for name in (
            'EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP',
            'FixedRandomFeedback', 'KappaFilter', 'PresynapticTrace',
        ):
            assert hasattr(pkg, name), f'missing export: {name}'

    def test_top_level_exports(self):
        import braintrace
        for name in ('EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP'):
            assert hasattr(braintrace, name), f'missing top-level export: {name}'
            assert name in braintrace.__all__
