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
import brainevent
import brainstate
import jax
import jax.numpy as jnp

import braintrace


class Test_MatMulOp:
    def test1(self):
        fn = braintrace.MatMulOp(weight_fn=jnp.abs)
        fn = braintrace.MatMulOp()
        x = brainstate.random.rand(10)
        w = {'weight': brainstate.random.randn(10, 20)}
        y1 = fn(x, w)
        dy = brainstate.random.randn(20)

        dw1 = fn.yw_to_w(dy, w)
        y2, f_vjp = jax.vjp(fn.xw_to_y, x, w)
        dx, dw2 = f_vjp(dy)

        assert jnp.allclose(y1, y2)
        print(dw1['weight'].shape)
        print(dw2['weight'].shape)


class Test_SpMatMulOp:
    def test1(self):
        mask = brainstate.random.rand(10, 20) > 0.5
        weight = jnp.where(mask, brainstate.random.rand(10, 20), 0.)
        csr = brainevent.CSR.fromdense(weight)

        fn = braintrace.SpMatMulOp(csr, weight_fn=jnp.abs)
        x = brainstate.random.rand(10)
        y1 = fn(x, {'weight': csr.data})
        dy = brainstate.random.randn(20)

        dw1 = fn.yw_to_w(dy, {'weight': csr.data})
        y2, f_vjp = jax.vjp(fn.xw_to_y, x, {'weight': csr.data})
        dx, dw2 = f_vjp(dy)

        assert jnp.allclose(y1, y2)
        print(dw1['weight'].shape)
        print(dw2['weight'].shape)
