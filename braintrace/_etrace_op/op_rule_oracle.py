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

"""Ground-truth checks for ETP-primitive rules (test support).

The defining property of a primitive's ``xy_to_dw`` rule is that it equals the
JAX vector-Jacobian product of that primitive's ``impl`` with respect to the
trainable weights, evaluated at the learning-signal cotangent ``hidden_dim``.
"""

import jax
import jax.numpy as jnp
import numpy as np


def xy_to_dw_and_vjp(*, rule, impl, x, hidden_dim, weights, params=None):
    """Return ``(rule_dw, vjp_dw)`` — the rule's weight gradient and the JAX vjp.

    ``impl`` must be a single-argument callable of the weights tree (closing over
    ``x`` and any static params), returning the primitive output ``y``.
    """
    params = params or {}
    rule_dw = rule(x, hidden_dim, weights, **params)
    _, vjp_fn = jax.vjp(impl, weights)
    vjp_dw = vjp_fn(hidden_dim)[0]
    return rule_dw, vjp_dw


def assert_xy_to_dw_matches_vjp(*, rule, impl, x, hidden_dim, weights, params=None, atol=1e-5, keys=None):
    """Assert the rule's weight gradient equals vjp(impl)(hidden_dim).

    ``keys`` restricts the comparison to a subset of weight leaves. This matters
    for primitives that *defer* part of a gradient's reduction (e.g. conv defers
    the spatial summation of the bias gradient) — those leaves are checked
    separately by the caller rather than compared element-wise here.
    """
    rule_dw, vjp_dw = xy_to_dw_and_vjp(
        rule=rule, impl=impl, x=x, hidden_dim=hidden_dim, weights=weights, params=params
    )
    compare = list(rule_dw.keys()) if keys is None else list(keys)
    if keys is None:
        assert set(rule_dw.keys()) == set(vjp_dw.keys()), (
            f"key mismatch: rule={set(rule_dw.keys())} vjp={set(vjp_dw.keys())}"
        )
    for key in compare:
        a = jnp.asarray(rule_dw[key])
        e = jnp.asarray(vjp_dw[key])
        np.testing.assert_allclose(a, e, atol=atol, err_msg=f"xy_to_dw mismatch on {key!r}")
