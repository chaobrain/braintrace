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

"""Tests for the convolution ETP primitive and the :func:`conv` API.

Coverage:

* Forward correctness — agrees with ``jax.lax.conv_general_dilated``
  under SAME and VALID padding, with and without bias.
* The ``etp_conv_p`` primitive appears in the jaxpr and carries the
  expected static parameters.
* brainunit support — quantities with units multiply correctly.
* JAX rules — jit / grad work.
* Four ETP rules — the ``conv`` rules use a VJP-based ``xy_to_dw`` that
  must match a hand-written JAX VJP; the init fns return arrays of the
  documented shape.
"""



from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

import braintrace
from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    conv,
    etp_conv_p,
)

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# Standard NCHW-style conv1d: input (batch, channel, length), kernel
# (out_channel, in_channel, kw).
def _ref_conv(x, kernel, **kw):
    return jax.lax.conv_general_dilated(x, kernel, **kw)


_BASE_CONV_KW = dict(
    window_strides=(1,),
    padding='SAME',
    lhs_dilation=(1,),
    rhs_dilation=(1,),
    feature_group_count=1,
    batch_group_count=1,
    dimension_numbers=None,
)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForwardCorrectness:

    def test_same_padding_no_bias(self):
        x = jnp.ones((1, 3, 8))  # (batch, in_ch, L)
        k = jnp.arange(36.0).reshape(4, 3, 3)  # (out_ch, in_ch, kw)
        out = conv(x, k)
        ref = _ref_conv(x, k, **_BASE_CONV_KW)
        np.testing.assert_allclose(out, ref)

    def test_valid_padding(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k, padding='VALID')
        ref_kw = dict(_BASE_CONV_KW)
        ref_kw['padding'] = 'VALID'
        ref = _ref_conv(x, k, **ref_kw)
        np.testing.assert_allclose(out, ref)

    def test_with_bias(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        b = jnp.arange(4.0)
        # bias broadcasts on the channel dim
        out = conv(x, k, b.reshape(1, 4, 1))
        ref = _ref_conv(x, k, **_BASE_CONV_KW) + b.reshape(1, 4, 1)
        np.testing.assert_allclose(out, ref)

    def test_strides(self):
        x = jnp.ones((1, 3, 16))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k, strides=(2,), padding='VALID')
        ref_kw = dict(_BASE_CONV_KW)
        ref_kw['window_strides'] = (2,)
        ref_kw['padding'] = 'VALID'
        ref = _ref_conv(x, k, **ref_kw)
        np.testing.assert_allclose(out, ref)


# ---------------------------------------------------------------------------
# Primitive bind + params
# ---------------------------------------------------------------------------

class TestPrimitiveAndParams:

    def test_jaxpr_contains_etp_conv(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(lambda x, k: conv(x, k))(x, k)
        assert any(eqn.primitive is etp_conv_p for eqn in jaxpr.jaxpr.eqns)

    def test_has_bias_true_when_bias_supplied(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        b = jnp.zeros((1, 4, 1))
        jaxpr = jax.make_jaxpr(lambda x, k, b: conv(x, k, b))(x, k, b)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['has_bias'] is True

    def test_has_bias_false_when_bias_omitted(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(lambda x, k: conv(x, k))(x, k)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['has_bias'] is False

    def test_strides_propagate(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(
            lambda x, k: conv(x, k, strides=(2,))
        )(x, k)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['strides'] == (2,)


# ---------------------------------------------------------------------------
# brainunit
# ---------------------------------------------------------------------------

class TestBrainunit:

    def test_unitless(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k)
        assert not isinstance(out, u.Quantity)

    def test_unit_multiplication(self):
        x = jnp.ones((1, 3, 8)) * u.mV
        k = jnp.ones((4, 3, 3))
        out = conv(x, k)
        ref = _ref_conv(jnp.ones((1, 3, 8)), k, **_BASE_CONV_KW)
        np.testing.assert_allclose(u.get_mantissa(out), ref)


# ---------------------------------------------------------------------------
# JAX rules
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = jax.jit(conv)(x, k)
        ref = _ref_conv(x, k, **_BASE_CONV_KW)
        np.testing.assert_allclose(out, ref)

    def test_grad_wrt_kernel(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        # Compare to grad through hand-written conv_general_dilated.
        gk_etp = jax.grad(lambda k_: conv(x, k_).sum())(k)
        gk_ref = jax.grad(
            lambda k_: _ref_conv(x, k_, **_BASE_CONV_KW).sum()
        )(k)
        np.testing.assert_allclose(gk_etp, gk_ref)


# ---------------------------------------------------------------------------
# ETP rules
# ---------------------------------------------------------------------------

class TestConvEtpRules:

    def test_yw_to_w_broadcasts_hidden_no_bias(self):
        rule = ETP_RULES_YW_TO_W[etp_conv_p]
        # Simulate the scan/etrace-update call (batch prefix retained).
        # 1-D NHC-HIO conv: kernel (H_k=3, in_ch=4, out_ch=4), output (N, H_out=6, C=4).
        # In the update path hidden_dim has the full batched output shape
        # (batch, H_out, out_ch) before spatial reduction.
        # trace['weight'] = (batch, H_k, in_ch, out_ch) — batch prefix present.
        # dimension_numbers and strides must be supplied so _conv_layout can derive
        # the spatial rank and kernel out-channel axis.
        batch = 1
        out_ch = 4
        H_out = 6
        # hidden_dim has full batched-output shape: (batch, H_out, out_ch)
        hidden = jnp.ones((batch, H_out, out_ch)) * jnp.arange(1, 5)  # (1, 6, 4)
        w_trace = jnp.ones((batch, 3, 4, out_ch))  # (1, H_k, in_ch, out_ch)
        trace = {'weight': w_trace}
        params = dict(has_bias=False, strides=(1,), dimension_numbers=('NHC', 'HIO', 'NHC'))
        out = rule(hidden, trace, **params)
        assert out['weight'].shape == (1, 3, 4, 4)
        assert 'bias' not in out
        # For NHC/HIO: has_batch_prefix=True (ndim=3 == n_spatial+2=3).
        # Spatial is summed: hd_reduced = hidden.sum(axis=1) → (1, 4).
        # w_out_axis = kernel_out_axis + 1 = 2+1 = 3 (O is at index 2 in HIO, +1 for batch).
        # target_shape = [1, 1, 1, 4] broadcast against (1, 3, 4, 4).
        hd_reduced = hidden.sum(axis=1)  # (1, 4) — sum over H_out
        expected = w_trace * hd_reduced[:, None, None, :]  # (1, 3, 4, 4)
        np.testing.assert_allclose(out['weight'], expected)

    def test_yw_to_w_broadcasts_hidden_with_bias(self):
        rule = ETP_RULES_YW_TO_W[etp_conv_p]
        # Simulate post-n_state-vmap shapes for 1-D conv NHC-HIO with spatial output.
        # hidden_dim = (batch=1, H_out=6, out_ch=4) — spatial output retained.
        # trace['weight'] = (batch=1, H_k=3, in_ch=4, out_ch=4).
        # trace['bias']   = (batch=1, H_out=6, out_ch=4)  ← same as y output (per-position).
        # dimension_numbers and strides must be supplied so _conv_layout can derive
        # the spatial rank and channel-axis position.
        batch = 1
        hidden = jnp.ones((batch, 6, 4))  # (1, H_out, out_ch)
        w_trace = jnp.ones((batch, 3, 4, 4))  # (1, H_k, in_ch, out_ch)
        b_trace = jnp.ones((batch, 6, 4)) * 2.0  # (1, H_out, out_ch) — per-position trace
        trace = {'weight': w_trace, 'bias': b_trace}
        params = dict(has_bias=True, strides=(1,), dimension_numbers=('NHC', 'HIO', 'NHC'))
        out = rule(hidden, trace, **params)
        assert 'weight' in out
        assert 'bias' in out
        assert out['weight'].shape == (1, 3, 4, 4)
        # bias result: sum over H_out of (b_trace * hidden) → (1, out_ch)
        assert out['bias'].shape == (1, 4)
        # bias update: elementwise product then sum over spatial (axis 1)
        expected_bias = jnp.sum(b_trace * hidden, axis=1)  # sum over H_out → (1, 4)
        np.testing.assert_allclose(out['bias'], expected_bias)

    def test_xy_to_dw_matches_jax_vjp(self):
        """The rule forwards its kwargs straight to
        ``jax.lax.conv_general_dilated`` (minus ``has_bias``); supply the
        ``window_strides`` / dilation kwargs that ``conv_general_dilated``
        natively expects."""
        rule = ETP_RULES_XY_TO_DW[etp_conv_p]
        x = jnp.ones((1, 3, 8))
        k = jnp.arange(36.0).reshape(4, 3, 3)
        ref_y_shape = _ref_conv(x, k, **_BASE_CONV_KW).shape
        hidden = jnp.ones(ref_y_shape)
        weights = {'weight': k}
        dk_dict = rule(x, hidden, weights, has_bias=False, **_BASE_CONV_KW)
        _, vjp_fn = jax.vjp(
            lambda k_: _ref_conv(x, k_, **_BASE_CONV_KW), k
        )
        ref_dk = vjp_fn(hidden)[0]
        np.testing.assert_allclose(dk_dict['weight'], ref_dk)
        assert 'bias' not in dk_dict

    def test_xy_to_dw_with_bias(self):
        """xy_to_dw returns both 'weight' and 'bias' gradients when has_bias=True.

        The bias 'gradient' is the cotangent (hidden_dim) itself — same shape as y.
        No spatial summation: that is deferred to _conv_yw_to_w.
        """
        rule = ETP_RULES_XY_TO_DW[etp_conv_p]
        x = jnp.ones((1, 3, 8))
        k = jnp.arange(36.0).reshape(4, 3, 3)
        b = jnp.ones(4)
        ref_y_shape = _ref_conv(x, k, **_BASE_CONV_KW).shape
        hidden = jnp.ones(ref_y_shape)  # (1, 4, 8) in NCH layout
        weights = {'weight': k, 'bias': b}
        dk_dict = rule(x, hidden, weights, has_bias=True, **_BASE_CONV_KW)
        assert 'weight' in dk_dict
        assert 'bias' in dk_dict
        # bias 'trace' = hidden_dim itself (per-position, no summation)
        assert dk_dict['bias'].shape == hidden.shape
        np.testing.assert_allclose(dk_dict['bias'], hidden)

    def test_init_drtrl_shape_no_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        weight_vars = {'weight': _fake_var((4, 3, 3))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (1, 4, 3, 3, 2)
        assert 'bias' not in out

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        # bias has shape (4,) but the trace stores per-position ∂h/∂b
        weight_vars = {'weight': _fake_var((4, 3, 3)), 'bias': _fake_var((4,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out['weight'].shape == (1, 4, 3, 3, 2)
        # bias trace: (batch, *y_shape[1:], n_state) = (1, 4, 8, 2)
        assert out['bias'].shape == (1, 4, 8, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        weight_vars = {'weight': _fake_var((4, 3, 3))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (1, 4, 8, 2)


# ---------------------------------------------------------------------------
# Bias-gradient correctness (D-RTRL vs BPTT)
# ---------------------------------------------------------------------------

class TestConvBiasGradient:
    """D-RTRL gradient correctness for etp_conv_p with a bias vector."""

    def test_drtrl_conv_grad_matches_bptt(self):
        """D-RTRL dkernel and dbias must match BPTT for one recurrent step."""
        kernel_init = jnp.ones((3, 4, 4)) * 0.05  # (H, in, out) for 1D conv NHC-HIO
        bias_init = jnp.ones((4,)) * 0.1

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': kernel_init, 'bias': bias_init}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, 6, 4)))

            def update(self, x):
                k = self.p.value['weight']
                b = self.p.value['bias']
                y = braintrace.conv(
                    self.h.value, k, b,
                    strides=(1,), padding='SAME',
                    dimension_numbers=('NHC', 'HIO', 'NHC'),
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 6, 4)))

        x = jnp.ones((1, 6, 4)) * 0.3

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # grads_etrace is keyed by path; cell.p is a dict-valued ParamState
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, 6, 4))
            y = jax.lax.conv_general_dilated(
                lhs=h, rhs=params['weight'],
                window_strides=(1,), padding='SAME',
                dimension_numbers=('NHC', 'HIO', 'NHC'),
            ) + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias': cell.p.value['bias'],
        })

        # Non-zero sanity check for bias gradient
        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        np.testing.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                                   err_msg='D-RTRL dkernel does not match BPTT')
        np.testing.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                                   err_msg='D-RTRL dbias does not match BPTT (was it zero?)')


class TestConv2dBiasGradient:
    """Bias gradient matches BPTT oracle for 2D conv (default NHWC / HWIO layout)."""

    def _run_test(self, spatial_h, spatial_w, kernel_h, kernel_w, in_ch, out_ch,
                  strides, padding, x_val=0.3):
        """Shared helper: build a tiny 2D recurrent cell and compare D-RTRL vs BPTT."""
        # Kernel shape: (Hk, Wk, in_ch, out_ch)  — HWIO layout
        kernel_init = jnp.ones((kernel_h, kernel_w, in_ch, out_ch)) * 0.05
        bias_init = jnp.ones((out_ch,)) * 0.1

        # Compute output spatial size for SAME padding (input size unchanged).
        # For VALID: floor((H - Hk) / stride + 1).
        if padding == 'SAME':
            out_h = -(-spatial_h // strides[0])  # ceil division
            out_w = -(-spatial_w // strides[1])
        else:  # VALID
            out_h = (spatial_h - kernel_h) // strides[0] + 1
            out_w = (spatial_w - kernel_w) // strides[1] + 1

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': kernel_init, 'bias': bias_init}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, out_h, out_w, out_ch)))

            def update(self, x):
                k = self.p.value['weight']
                b = self.p.value['bias']
                y = braintrace.conv(
                    self.h.value, k, b,
                    strides=strides, padding=padding,
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, out_h, out_w, out_ch)))

        x = jnp.ones((1, out_h, out_w, out_ch)) * x_val

        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, out_h, out_w, out_ch))
            y = jax.lax.conv_general_dilated(
                lhs=h, rhs=params['weight'],
                window_strides=strides, padding=padding,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            ) + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias': cell.p.value['bias'],
        })

        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        np.testing.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                                   err_msg='D-RTRL dkernel does not match BPTT')
        np.testing.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                                   err_msg='D-RTRL dbias does not match BPTT (was it zero?)')

    def test_conv2d_default_layout(self):
        """2D conv, default NHWC/HWIO layout, 3x3 kernel, stride 1, SAME padding."""
        self._run_test(
            spatial_h=6, spatial_w=6,
            kernel_h=3, kernel_w=3,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='SAME',
        )

    def test_conv2d_with_valid_padding(self):
        """2D conv, VALID padding, 1x1 kernel — preserves spatial dims without growth.

        A 1x1 kernel with VALID padding is a valid non-trivial test: every output
        position directly maps to one input position (no spatial reduction), so the
        hidden state remains stable in a recurrent cell.  This exercises the VALID
        padding code path and the 2D kernel handling simultaneously.
        """
        self._run_test(
            spatial_h=4, spatial_w=4,
            kernel_h=1, kernel_w=1,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='VALID',
        )

    def test_conv2d_rectangular_kernel(self):
        """2D conv with a non-square 3x1 kernel (horizontal strip)."""
        self._run_test(
            spatial_h=6, spatial_w=6,
            kernel_h=3, kernel_w=1,
            in_ch=4, out_ch=4,
            strides=(1, 1), padding='SAME',
        )


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.conv is conv


class TestConvKernelFnBiasFn:

    def test_forward_applies_kernel_fn(self):
        x = brainstate.random.randn(8, 3, 16)
        k = brainstate.random.randn(4, 3, 5)
        out = braintrace.conv(x, k, strides=(1,), padding='SAME', kernel_fn=lambda w: w ** 2)
        import jax
        ref = jax.lax.conv_general_dilated(x, k ** 2, window_strides=(1,), padding='SAME')
        np.testing.assert_allclose(out, ref, atol=1e-4)

    def test_kernel_fn_xy_to_dw_matches_vjp(self):
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_conv_p
        import jax
        rule = ETP_RULES_XY_TO_DW[etp_conv_p]
        x = brainstate.random.randn(2, 3, 10)
        k = brainstate.random.randn(4, 3, 5)
        hidden = brainstate.random.randn(2, 4, 10)
        params = dict(has_bias=False, strides=(1,), padding='SAME', lhs_dilation=None,
                      rhs_dilation=None, feature_group_count=1, batch_group_count=1,
                      dimension_numbers=None, kernel_fn=lambda w: w ** 2, bias_fn=None)
        dw = rule(x, hidden, {'weight': k}, **params)

        def fwd(w):
            return jax.lax.conv_general_dilated(x, w ** 2, window_strides=(1,), padding='SAME')

        _, vjp = jax.vjp(fwd, k)
        np.testing.assert_allclose(dw['weight'], vjp(hidden)[0], atol=1e-4)

    def test_bias_fn_xy_to_dw_factor(self):
        """Deferred bias trace carries bias_fn'(b) per channel (NCH: channel axis=1)."""
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_conv_p
        rule = ETP_RULES_XY_TO_DW[etp_conv_p]
        x = brainstate.random.randn(2, 3, 10)
        k = brainstate.random.randn(4, 3, 5)
        b = brainstate.random.randn(4)
        hidden = brainstate.random.randn(2, 4, 10)
        params = dict(has_bias=True, strides=(1,), padding='SAME', lhs_dilation=None,
                      rhs_dilation=None, feature_group_count=1, batch_group_count=1,
                      dimension_numbers=None, kernel_fn=None, bias_fn=lambda bb: bb ** 2)
        out = rule(x, hidden, {'weight': k, 'bias': b}, **params)
        # bias_fn'(b) = 2*b, broadcast along channel axis=1 of the (batch, ch, spatial) trace.
        expected = hidden * (2.0 * b).reshape(1, 4, 1)
        np.testing.assert_allclose(out['bias'], expected, atol=1e-4)
