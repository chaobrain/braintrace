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
        import braintrace._algorithm as pkg
        for name in (
                'EProp', 'OSTLRecurrent', 'OSTLFeedforward',
                'FixedRandomFeedback', 'KappaFilter',
        ):
            assert hasattr(pkg, name), f'missing export: {name}'

    def test_top_level_exports(self):
        import braintrace
        for name in ('EProp', 'OSTLRecurrent', 'OSTLFeedforward'):
            assert hasattr(braintrace, name), f'missing top-level export: {name}'
            assert name in braintrace.__all__

    def test_removed_algorithms_are_gone(self):
        """OTTT / OSTTP / OTPE were removed in 0.2.5 (see docs/specs roadmap).

        They whitelisted dense-matmul primitives and were not model-agnostic, so
        they no longer belong in a general framework. Their coordinates remain
        reachable through the axis configuration space.
        """
        import braintrace
        import braintrace._algorithm as pkg
        for name in ('OTTT', 'OSTTP', 'OTPE', 'PresynapticTrace'):
            assert not hasattr(braintrace, name), f'{name} should have been removed'
            assert name not in braintrace.__all__
            assert not hasattr(pkg, name), f'{name} should have been removed'
