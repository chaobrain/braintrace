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

import jax.numpy as jnp
from jax import jit, make_jaxpr, lax

from braintrace._compatible_imports import (
    is_jit_primitive, is_scan_primitive, is_while_primitive,
    is_cond_primitive
)


class TestPrimitive:
    def test_jit(self):
        @jit
        def jit_function(x, y):
            return x ** 2 + jnp.sin(y)

        # Note: make_jaxpr on a jitted function shows the same jaxpr
        jaxpr_jit = make_jaxpr(jit_function)(2.0, 1.0)
        assert is_jit_primitive(jaxpr_jit.eqns[0])

    def test_scan(self):
        print("3. make_jaxpr with lax.scan:")

        def scan_step(carry, x):
            return carry + x, carry * x

        def scan_function(init, xs):
            return lax.scan(scan_step, init, xs)

        # Create sample data
        init_val = 1.0
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        jaxpr_scan = make_jaxpr(scan_function)(init_val, xs)
        assert is_scan_primitive(jaxpr_scan.eqns[0])

    def test_while(self):
        def while_cond(carry):
            i, x = carry
            return i < 5

        def while_body(carry):
            i, x = carry
            return i + 1, x * 2

        def while_function(init_carry):
            return lax.while_loop(while_cond, while_body, init_carry)

        init_carry = (0, 1.0)
        jaxpr_while = make_jaxpr(while_function)(init_carry)
        assert is_while_primitive(jaxpr_while.eqns[0])

    def test_cond(self):
        def true_branch(x):
            return x * 2

        def false_branch(x):
            return x + 1

        def cond_function(pred, x):
            return lax.cond(pred, true_branch, false_branch, x)

        jaxpr_cond = make_jaxpr(cond_function)(True, 5.0)
        assert is_cond_primitive(jaxpr_cond.eqns[-1])

    def test_fori_loop(self):
        def branch_0(x):
            return x * 2

        def branch_1(x):
            return x + 10

        def branch_2(x):
            return x ** 2

        def switch_function(index, x):
            return lax.switch(index, [branch_0, branch_1, branch_2], x)

        jaxpr_switch = make_jaxpr(switch_function)(1, 3.0)
        assert is_cond_primitive(jaxpr_switch.eqns[-1])
