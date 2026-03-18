# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for ETP primitives and user-facing functions."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from braintrace.etp import (
    matmul,
    element_wise,
    etp_matmul_p,
    etp_elemwise_p,
    etp_conv_p,
    is_etp_primitive,
)


# ==============================================================================
# etp_matmul tests
# ==============================================================================

class TestETPMatmul:
    """Tests for the etp_matmul primitive and function."""

    def test_forward_no_bias(self):
        x = jnp.ones((3, 4))
        w = jnp.ones((4, 5))
        y = matmul(x, w)
        expected = x @ w
        npt.assert_allclose(y, expected)
        assert y.shape == (3, 5)

    def test_forward_with_bias(self):
        x = jnp.ones((3, 4))
        w = jnp.ones((4, 5))
        b = jnp.ones(5) * 2.0
        y = matmul(x, w, bias=b)
        expected = x @ w + b
        npt.assert_allclose(y, expected)

    def test_jit(self):
        x = jnp.ones((3, 4))
        w = jnp.ones((4, 5))
        y = jax.jit(matmul)(x, w)
        npt.assert_allclose(y, x @ w)

    def test_grad_wrt_input(self):
        x = jax.random.normal(jax.random.key(0), (3, 4))
        w = jax.random.normal(jax.random.key(1), (4, 5))
        grad_x = jax.grad(lambda x: matmul(x, w).sum())(x)
        expected = jax.grad(lambda x: (x @ w).sum())(x)
        npt.assert_allclose(grad_x, expected, atol=1e-6)

    def test_grad_wrt_weight(self):
        x = jax.random.normal(jax.random.key(0), (3, 4))
        w = jax.random.normal(jax.random.key(1), (4, 5))
        grad_w = jax.grad(lambda w: matmul(x, w).sum())(w)
        expected = jax.grad(lambda w: (x @ w).sum())(w)
        npt.assert_allclose(grad_w, expected, atol=1e-6)

    def test_grad_wrt_bias(self):
        x = jax.random.normal(jax.random.key(0), (3, 4))
        w = jax.random.normal(jax.random.key(1), (4, 5))
        b = jax.random.normal(jax.random.key(2), (5,))
        grad_b = jax.grad(lambda b: matmul(x, w, bias=b).sum())(b)
        expected = jax.grad(lambda b: (x @ w + b).sum())(b)
        npt.assert_allclose(grad_b, expected, atol=1e-6)

    def test_vmap_over_x(self):
        batch_x = jax.random.normal(jax.random.key(0), (8, 3, 4))
        w = jax.random.normal(jax.random.key(1), (4, 5))
        y = jax.vmap(lambda xi: matmul(xi, w))(batch_x)
        expected = jax.vmap(lambda xi: xi @ w)(batch_x)
        npt.assert_allclose(y, expected, atol=1e-6)
        assert y.shape == (8, 3, 5)

    def test_make_jaxpr_contains_etp_primitive(self):
        x = jnp.ones((3, 4))
        w = jnp.ones((4, 5))
        jaxpr = jax.make_jaxpr(lambda x: matmul(x, w))(x)
        primitives = [eqn.primitive for eqn in jaxpr.jaxpr.eqns]
        assert etp_matmul_p in primitives, (
            f'etp_matmul_p not found in jaxpr primitives: '
            f'{[p.name for p in primitives]}'
        )

    def test_jvp(self):
        x = jax.random.normal(jax.random.key(0), (3, 4))
        w = jax.random.normal(jax.random.key(1), (4, 5))
        dx = jax.random.normal(jax.random.key(2), (3, 4))

        primal, tangent = jax.jvp(
            lambda x: matmul(x, w), (x,), (dx,)
        )
        expected_primal = x @ w
        expected_tangent = dx @ w
        npt.assert_allclose(primal, expected_primal, atol=1e-6)
        npt.assert_allclose(tangent, expected_tangent, atol=1e-6)


# ==============================================================================
# etp_elemwise tests
# ==============================================================================

class TestETPElemwise:
    """Tests for the etp_elemwise primitive and function."""

    def test_identity(self):
        w = jnp.array([1.0, 2.0, 3.0])
        y = element_wise(w)
        npt.assert_allclose(y, w)

    def test_with_fn(self):
        w = jnp.array([1.0, 2.0, 3.0])
        y = element_wise(w, fn=jax.nn.softplus)
        expected = jax.nn.softplus(w)
        npt.assert_allclose(y, expected, atol=1e-6)

    def test_grad(self):
        w = jax.random.normal(jax.random.key(0), (5,))
        grad_w = jax.grad(lambda w: element_wise(w, fn=jax.nn.softplus).sum())(w)
        expected = jax.grad(lambda w: jax.nn.softplus(w).sum())(w)
        npt.assert_allclose(grad_w, expected, atol=1e-6)

    def test_jit(self):
        w = jnp.array([1.0, 2.0, 3.0])
        y = jax.jit(lambda w: element_wise(w, fn=lambda x: x * 2))(w)
        npt.assert_allclose(y, w * 2)

    def test_vmap(self):
        w = jax.random.normal(jax.random.key(0), (8, 5))
        y = jax.vmap(lambda wi: element_wise(wi))(w)
        npt.assert_allclose(y, w)

    def test_make_jaxpr_contains_etp_primitive(self):
        w = jnp.ones(5)
        jaxpr = jax.make_jaxpr(lambda w: element_wise(w))(w)
        primitives = [eqn.primitive for eqn in jaxpr.jaxpr.eqns]
        assert etp_elemwise_p in primitives

    def test_jvp(self):
        w = jax.random.normal(jax.random.key(0), (5,))
        dw = jax.random.normal(jax.random.key(1), (5,))
        primal, tangent = jax.jvp(
            lambda w: element_wise(w, fn=jnp.tanh), (w,), (dw,)
        )
        expected_primal = jnp.tanh(w)
        expected_tangent = dw * (1 - jnp.tanh(w) ** 2)  # tanh'
        npt.assert_allclose(primal, expected_primal, atol=1e-6)
        npt.assert_allclose(tangent, expected_tangent, atol=1e-5)


# ==============================================================================
# is_etp_primitive
# ==============================================================================

class TestIsETPPrimitive:
    def test_etp_primitives(self):
        assert is_etp_primitive(etp_matmul_p)
        assert is_etp_primitive(etp_elemwise_p)
        assert is_etp_primitive(etp_conv_p)

    def test_non_etp_primitives(self):
        assert not is_etp_primitive(jax.lax.add_p)
        assert not is_etp_primitive(jax.lax.mul_p)


# ==============================================================================
# Composability tests
# ==============================================================================

class TestComposability:
    """Test that ETP primitives compose with JAX transformations."""

    def test_jit_grad(self):
        """jax.jit(jax.grad(f)) where f uses etp_matmul."""
        x = jax.random.normal(jax.random.key(0), (4,))
        w = jax.random.normal(jax.random.key(1), (4, 3))

        @jax.jit
        def f(x):
            return matmul(x, w).sum()

        grad_x = jax.grad(f)(x)
        expected = jax.grad(lambda x: (x @ w).sum())(x)
        npt.assert_allclose(grad_x, expected, atol=1e-6)

    def test_vmap_grad(self):
        """jax.vmap(jax.grad(f)) where f uses etp_matmul."""
        batch_x = jax.random.normal(jax.random.key(0), (8, 4))
        w = jax.random.normal(jax.random.key(1), (4, 3))

        grad_fn = jax.vmap(jax.grad(lambda x: matmul(x, w).sum()))
        grads = grad_fn(batch_x)

        expected = jax.vmap(jax.grad(lambda x: (x @ w).sum()))(batch_x)
        npt.assert_allclose(grads, expected, atol=1e-6)

    def test_grad_vmap(self):
        """jax.grad of vmapped function using etp_matmul."""
        batch_x = jax.random.normal(jax.random.key(0), (8, 4))
        w = jax.random.normal(jax.random.key(1), (4, 3))

        def f(w):
            return jax.vmap(lambda x: matmul(x, w))(batch_x).sum()

        grad_w = jax.grad(f)(w)
        expected = jax.grad(lambda w: jax.vmap(lambda x: x @ w)(batch_x).sum())(w)
        npt.assert_allclose(grad_w, expected, atol=1e-5)

    def test_weight_fn_before_primitive(self):
        """
        Test that weight_fn applied before the primitive works correctly
        with grad — JAX handles the chain rule through weight_fn.
        """
        x = jax.random.normal(jax.random.key(0), (4,))
        raw_w = jax.random.normal(jax.random.key(1), (4, 3))

        # Apply weight_fn outside the primitive
        def f(raw_w):
            w = jax.nn.softplus(raw_w)  # weight_fn
            return matmul(x, w).sum()

        grad_raw_w = jax.grad(f)(raw_w)

        # Reference: same thing without primitive
        def f_ref(raw_w):
            w = jax.nn.softplus(raw_w)
            return (x @ w).sum()

        expected = jax.grad(f_ref)(raw_w)
        npt.assert_allclose(grad_raw_w, expected, atol=1e-5)

    def test_mask_before_primitive(self):
        """Test weight mask applied before the primitive."""
        x = jax.random.normal(jax.random.key(0), (4,))
        w = jax.random.normal(jax.random.key(1), (4, 3))
        mask = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])

        def f(w):
            return matmul(x, w * mask).sum()

        grad_w = jax.grad(f)(w)
        expected = jax.grad(lambda w: (x @ (w * mask)).sum())(w)
        npt.assert_allclose(grad_w, expected, atol=1e-5)
