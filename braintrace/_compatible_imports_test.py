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

import types

import jax.numpy as jnp
from jax import jit, make_jaxpr, lax

from braintrace._compatible_imports import (
    is_jit_primitive, is_scan_primitive, is_while_primitive,
    is_cond_primitive, scan_num_consts_carry, scan_params_add_ys,
)


def _one_const_one_carry_one_xs_scan_eqn():
    """A scan with exactly 1 const, 1 carry, 1 xs and 1 y."""

    def scan_function(w, init, xs):
        def step(c, x):
            return c + w + x, c * 2.0  # w=const, c=carry, x=xs, y=c*2

        return lax.scan(step, init, xs)

    jaxpr = make_jaxpr(scan_function)(2.0, 1.0, jnp.arange(4.0))
    return next(e for e in jaxpr.eqns if is_scan_primitive(e))


class TestScanConstsCarry:
    """`scan_num_consts_carry` across the JAX <0.11 and 0.11 scan encodings."""

    def test_real_scan_eqn(self):
        # Works on either encoding: old jax reads num_consts/num_carry params,
        # jax 0.11 derives them from the ft_in flattree.
        eqn = _one_const_one_carry_one_xs_scan_eqn()
        assert scan_num_consts_carry(eqn) == (1, 1)

    def test_old_jax_param_branch(self):
        # Exercise the pre-0.11 branch without an old jax installed by handing
        # the shim an eqn-like object whose params carry the legacy keys.
        fake = types.SimpleNamespace(params={'num_consts': 3, 'num_carry': 2})
        assert scan_num_consts_carry(fake) == (3, 2)


class TestScanAddYs:
    """`scan_params_add_ys` extends the ys arity only where the encoding needs it."""

    def test_zero_extra_is_identity(self):
        eqn = _one_const_one_carry_one_xs_scan_eqn()
        params = dict(eqn.params)
        assert scan_params_add_ys(params, 0) is params

    def test_no_ft_out_is_identity(self):
        # Legacy encoding (no ft_out): num_ys is implicit, params untouched.
        legacy = {'num_consts': 1, 'num_carry': 1, 'length': 4}
        assert scan_params_add_ys(legacy, 2) is legacy

    def test_extends_ft_out_when_present(self):
        eqn = _one_const_one_carry_one_xs_scan_eqn()
        params = dict(eqn.params)
        if 'ft_out' not in params:
            # Old jax: nothing to extend, identity contract holds.
            assert scan_params_add_ys(params, 2) is params
            return
        old_carry, old_ys = params['ft_out'].unpack()
        new_params = scan_params_add_ys(params, 2)
        new_carry, new_ys = new_params['ft_out'].unpack()
        assert len(new_params['ft_out']) == len(params['ft_out']) + 2
        assert len(new_carry) == len(old_carry)      # carry unchanged
        assert len(new_ys) == len(old_ys) + 2        # ys grew by 2


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
