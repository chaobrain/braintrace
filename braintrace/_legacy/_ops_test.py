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

"""Smoke tests for the v0.1.x op compatibility shims.

Coverage:
* every legacy op constructs and executes forward.
* ``MatMulOp.xw_to_y`` matches its raw forward.
* the utility shims (``general_y2w``, ``stop_param_gradients``) work.
* ``DeprecationWarning`` fires on package-root op access.
* every legacy op is re-exported at the package root.
"""



import warnings

import jax
import jax.numpy as jnp
import numpy as np

import braintrace
from braintrace._legacy import (
    ConvOp,
    ElemWiseOp,
    ETraceOp,
    LoraOp,
    MatMulOp,
    SpMatMulOp,
    general_y2w,
    stop_param_gradients,
)


# ---------------------------------------------------------------------------
# Op forward pass
# ---------------------------------------------------------------------------

class TestOpsForward:

    def test_matmul_op(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        w = {'weight': jnp.ones((4, 3))}
        y = op(x, w)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y, x @ w['weight'])

    def test_matmul_op_with_bias(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        b = jnp.arange(3.0)
        w = {'weight': jnp.ones((4, 3)), 'bias': b}
        y = op(x, w)
        np.testing.assert_allclose(y, x @ w['weight'] + b)

    def test_matmul_op_weight_fn(self):
        op = MatMulOp(weight_fn=lambda w: w * 2)
        x = jnp.ones((2, 4))
        w = {'weight': jnp.ones((4, 3))}
        y = op(x, w)
        np.testing.assert_allclose(y, (x @ w['weight']) * 2)

    def test_matmul_op_raw_matches_etp(self):
        op = MatMulOp()
        x = jnp.ones((2, 4))
        w = {'weight': jnp.arange(12.0).reshape(4, 3)}
        np.testing.assert_allclose(op.xw_to_y(x, w), op.raw_xw_to_y(x, w))

    def test_elemwise_op(self):
        op = ElemWiseOp(fn=lambda w: w * 3)
        w = jnp.ones((5,))
        y = op(w)
        np.testing.assert_allclose(y, 3.0)

    def test_lora_op(self):
        op = LoraOp(alpha=2.0)
        x = jnp.ones((2, 4))
        w = {'B': jnp.ones((4, 2)), 'A': jnp.ones((2, 3))}
        y = op(x, w)
        expected = 2.0 * x @ w['B'] @ w['A']
        np.testing.assert_allclose(y, expected)

    def test_conv_op(self):
        xinfo = jax.ShapeDtypeStruct((1, 4, 4, 1), jnp.float32)
        op = ConvOp(
            xinfo=xinfo,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x = jnp.ones((1, 4, 4, 1))
        w = {'weight': jnp.ones((3, 3, 1, 2))}
        y = op(x, w)
        assert y.shape == (1, 4, 4, 2)

    def test_sparse_op(self):
        import brainevent
        dense = jnp.eye(3) * 2.0
        csr = brainevent.CSR.fromdense(dense)
        op = SpMatMulOp(sparse_mat=csr)
        x = jnp.arange(3.0)
        w = {'weight': csr.data}
        y = op(x, w)
        np.testing.assert_allclose(y, x @ dense)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_stop_param_gradients_context(self):
        # Pure API compat; just verify the context manager works.
        with stop_param_gradients(True):
            pass
        with stop_param_gradients(False):
            pass

    def test_general_y2w(self):
        def xw2y(x, w):
            return x @ w

        x = jnp.ones((4,))
        w = jnp.ones((4, 3))
        y = jnp.ones((3,))
        w_like = general_y2w(xw2y, x, y, w)
        assert w_like.shape == w.shape


# ---------------------------------------------------------------------------
# Deprecation warnings
# ---------------------------------------------------------------------------

class TestDeprecationWarnings:
    # As of 0.2.0 the deprecation warning fires at package-root attribute access
    # (``braintrace.MatMulOp``) via ``braintrace.__getattr__``, not at construction
    # and not when importing from the private ``braintrace._legacy`` submodule.

    def test_matmul_op_access_warns(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            _ = braintrace.MatMulOp
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'MatMulOp' in str(w.message)
            for w in captured
        )

    def test_construction_does_not_warn(self):
        # The shim classes themselves no longer warn; construction is silent.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            MatMulOp()
        assert not any(
            issubclass(w.category, DeprecationWarning) for w in captured
        )


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------

class TestTopLevelExports:

    def test_op_classes_at_top_level(self):
        assert braintrace.ETraceOp is ETraceOp
        assert braintrace.MatMulOp is MatMulOp
        assert braintrace.ElemWiseOp is ElemWiseOp
        assert braintrace.ConvOp is ConvOp
        assert braintrace.SpMatMulOp is SpMatMulOp
        assert braintrace.LoraOp is LoraOp
