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

"""Separate-ParamState layout test.

When weight and bias live in separate ParamStates, ETP must route
gradients to both paths correctly.
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt

import braintrace


class TestSeparateParamStateBias:
    """D-RTRL gradient correctness when weight and bias are in distinct ParamStates."""

    def test_separate_weight_and_bias_grads(self):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.b = brainstate.ParamState(jnp.ones((4,)) * 0.2)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, self.b.value)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3

        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # --- BPTT reference ---
        def bptt_loss(w, b):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ w + b)
            return h.sum()

        dW_bptt, db_bptt = jax.grad(bptt_loss, (0, 1))(cell.w.value, cell.b.value)

        # grads_etrace is keyed by path: ('w',) -> array, ('b',) -> array
        leaves = jax.tree.leaves(grads_etrace)
        shapes = sorted(leaf.shape for leaf in leaves)
        assert shapes == [(4,), (4, 4)], (
            f'Expected two gradient leaves with shapes [(4,), (4, 4)], got {shapes}'
        )

        w_leaf = next(l for l in leaves if l.shape == (4, 4))
        b_leaf = next(l for l in leaves if l.shape == (4,))

        npt.assert_allclose(w_leaf, dW_bptt, atol=1e-5,
                            err_msg='D-RTRL dW does not match BPTT')
        npt.assert_allclose(b_leaf, db_bptt, atol=1e-5,
                            err_msg='D-RTRL db does not match BPTT (was it zero?)')
