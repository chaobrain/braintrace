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
import pytest
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
        # Simulate the scan/etrace-update (recurrence) call — batch retained.
        # 1-D NHC-HIO conv: kernel (H_k=3, in_ch=4, out_ch=4), output (N, H_out=6, C=4).
        # hidden_dim (the per-position D^t factor) has the full batched output
        # shape (batch, H_out, out_ch); the per-position kernel trace is
        # (batch, H_out, H_k, in_ch, out_ch). No spatial sums anywhere.
        batch = 1
        out_ch = 4
        H_out = 6
        brainstate.random.seed(0)
        hidden = brainstate.random.randn(batch, H_out, out_ch)
        w_trace = brainstate.random.randn(batch, H_out, 3, 4, out_ch)
        trace = {'weight': w_trace}
        params = dict(has_bias=False, strides=(1,), dimension_numbers=('NHC', 'HIO', 'NHC'))
        out = rule(hidden, trace, **params)
        assert out['weight'].shape == (1, H_out, 3, 4, out_ch)
        assert 'bias' not in out
        # Per-position multiply: hd[b, s, k] broadcast over (H_k, in_ch).
        expected = w_trace * hidden[:, :, None, None, :]
        np.testing.assert_allclose(out['weight'], expected)

    def test_yw_to_w_broadcasts_hidden_with_bias(self):
        rule = ETP_RULES_YW_TO_W[etp_conv_p]
        # Recurrence context for 1-D conv NHC-HIO.
        # hidden_dim = (batch=1, H_out=6, out_ch=4).
        # trace['weight'] = (batch=1, H_out=6, H_k=3, in_ch=4, out_ch=4).
        # trace['bias']   = (batch=1, H_out=6, out_ch=4)  ← y-shaped (per-position).
        batch = 1
        brainstate.random.seed(1)
        hidden = brainstate.random.randn(batch, 6, 4)
        w_trace = brainstate.random.randn(batch, 6, 3, 4, 4)
        b_trace = brainstate.random.randn(batch, 6, 4)
        trace = {'weight': w_trace, 'bias': b_trace}
        params = dict(has_bias=True, strides=(1,), dimension_numbers=('NHC', 'HIO', 'NHC'))
        out = rule(hidden, trace, **params)
        assert 'weight' in out
        assert 'bias' in out
        assert out['weight'].shape == (1, 6, 3, 4, 4)
        # Bias recurrence is a pure per-position multiply — NO spatial sum
        # (the old spatially-summed multiplier corrupted the recurrence for
        # T >= 2; the sum now lives in the solve rule).
        assert out['bias'].shape == (1, 6, 4)
        np.testing.assert_allclose(out['bias'], b_trace * hidden)

    def test_yw_to_w_default_nch_layout(self):
        """Default (NCH/OIH) layout: hd must be transposed to (batch, s, k)
        and its channel aligned to the kernel's out-channel axis (0 in OIH)."""
        rule = ETP_RULES_YW_TO_W[etp_conv_p]
        batch, c_out, length = 1, 3, 8
        brainstate.random.seed(2)
        hidden = brainstate.random.randn(batch, c_out, length)      # NCH
        w_trace = brainstate.random.randn(batch, length, c_out, 2, 3)  # (b, s, O, I, H)
        params = dict(has_bias=False, strides=(1,), dimension_numbers=None)
        out = rule(hidden, {'weight': w_trace}, **params)
        hd = jnp.transpose(hidden, (0, 2, 1))  # (b, s, k)
        expected = w_trace * hd[:, :, :, None, None]
        np.testing.assert_allclose(out['weight'], expected)

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
        y_var = _fake_var((1, 4, 8))  # NCH: spatial_out = (8,)
        weight_vars = {'weight': _fake_var((4, 3, 3))}
        eqn_params = dict(strides=(1,), dimension_numbers=None,
                          feature_group_count=1, batch_group_count=1)
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2,
                   eqn_params=eqn_params)
        # Per-position kernel trace: (batch, *spatial_out, *kernel, n_state).
        assert out['weight'].shape == (1, 8, 4, 3, 3, 2)
        assert 'bias' not in out

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        # bias has shape (4,) but the trace stores per-position ∂h/∂b
        weight_vars = {'weight': _fake_var((4, 3, 3)), 'bias': _fake_var((4,))}
        eqn_params = dict(strides=(1,), dimension_numbers=None,
                          feature_group_count=1, batch_group_count=1)
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2,
                   eqn_params=eqn_params)
        assert out['weight'].shape == (1, 8, 4, 3, 3, 2)
        # bias trace: (batch, *y_shape[1:], n_state) = (1, 4, 8, 2)
        assert out['bias'].shape == (1, 4, 8, 2)

    def test_init_drtrl_shape_channel_last(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 8, 3))   # NWC
        y_var = _fake_var((1, 8, 4))   # NWC: spatial_out = (8,)
        weight_vars = {'weight': _fake_var((3, 3, 4))}  # WIO
        eqn_params = dict(strides=(1,), dimension_numbers=('NWC', 'WIO', 'NWC'),
                          feature_group_count=1, batch_group_count=1)
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2,
                   eqn_params=eqn_params)
        assert out['weight'].shape == (1, 8, 3, 3, 4, 2)

    def test_init_drtrl_rejects_grouped_conv(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 4, 8))
        y_var = _fake_var((1, 4, 8))
        weight_vars = {'weight': _fake_var((4, 2, 3))}
        eqn_params = dict(strides=(1,), dimension_numbers=None,
                          feature_group_count=2, batch_group_count=1)
        with pytest.raises(NotImplementedError, match='pp_prop'):
            rule(x_var, y_var, weight_vars, num_hidden_state=1,
                 eqn_params=eqn_params)

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


# ---------------------------------------------------------------------------
# Bias broadcasts along the layout's channel axis (H3)
# ---------------------------------------------------------------------------

class TestConvBiasChannelAxis:
    """A ``(out_channels,)`` bias must broadcast along the layout's channel
    axis, not just whatever axis happens to be trailing. The default
    (``dimension_numbers=None``) layout is NCH-style, so the channel axis is
    at position 1 -- not the trailing spatial axis.
    """

    def test_default_layout_bias_broadcasts_channel_axis(self):
        # x: (batch=2, C_in=2, L=8); kernel: (C_out=4, C_in=2, kw=3); bias: (C_out=4,)
        x = jnp.arange(2 * 2 * 8, dtype=jnp.float32).reshape(2, 2, 8)
        k = jnp.arange(4 * 2 * 3, dtype=jnp.float32).reshape(4, 2, 3)
        b = jnp.arange(4.0)
        out = conv(x, k, b, strides=(1,), padding='SAME')
        ref = _ref_conv(x, k, **_BASE_CONV_KW) + b.reshape(1, -1, 1)
        np.testing.assert_allclose(out, ref)

    def test_default_layout_bias_hazard_when_spatial_equals_out_channels(self):
        """Regression guard for the silent-corruption hazard: spatial length
        ``L`` happens to equal ``C_out``. Before the fix this shape
        coincidence let the buggy trailing-axis broadcast succeed silently
        (no ``ValueError``) while producing the wrong numbers.
        """
        # L == C_out == 4
        x = jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4)
        k = jnp.arange(4 * 2 * 3, dtype=jnp.float32).reshape(4, 2, 3)
        b = jnp.arange(1.0, 5.0)
        out = conv(x, k, b, strides=(1,), padding='SAME')
        ref = _ref_conv(x, k, **_BASE_CONV_KW) + b.reshape(1, -1, 1)
        np.testing.assert_allclose(out, ref)
        # Must differ from the old (buggy) trailing-axis broadcast -- proves
        # this test actually discriminates correct vs. silently-corrupted output.
        buggy = _ref_conv(x, k, **_BASE_CONV_KW) + b.reshape(1, 1, -1)
        assert not np.allclose(out, buggy)

    def test_channel_last_layout_bias_still_works(self):
        """Channel-last (`NWC`/`WIO`/`NWC`) layout regression guard -- the
        channel axis is already trailing here, so this must keep working.
        """
        x = jnp.arange(2 * 8 * 2, dtype=jnp.float32).reshape(2, 8, 2)  # (N, W, C_in)
        k = jnp.arange(3 * 2 * 4, dtype=jnp.float32).reshape(3, 2, 4)  # (W, I, O)
        b = jnp.arange(4.0)
        dn = ('NWC', 'WIO', 'NWC')
        out = conv(x, k, b, strides=(1,), padding='SAME', dimension_numbers=dn)
        ref = _ref_conv(
            x, k, window_strides=(1,), padding='SAME', dimension_numbers=dn
        ) + b.reshape(1, 1, -1)
        np.testing.assert_allclose(out, ref)

    def test_grad_through_biased_default_layout_conv(self):
        """``jax.grad`` through a biased default-layout (NCH) conv must work.

        All standard rules (JVP / transpose) are auto-derived from
        ``_etp_conv_impl`` (see ``register_primitive``), so the channel-axis
        reshape added to fix the bias broadcast must itself be
        differentiable -- verify kernel and bias gradients match a
        hand-written reference that performs the same reshape.
        """
        x = jnp.ones((2, 2, 8))
        k = jnp.ones((4, 2, 3)) * 0.1
        b = jnp.arange(4.0)

        def loss(k_, b_):
            return conv(x, k_, b_, strides=(1,), padding='SAME').sum()

        gk, gb = jax.grad(loss, argnums=(0, 1))(k, b)

        def ref_loss(k_, b_):
            y = _ref_conv(x, k_, **_BASE_CONV_KW)
            return (y + b_.reshape(1, -1, 1)).sum()

        gk_ref, gb_ref = jax.grad(ref_loss, argnums=(0, 1))(k, b)
        np.testing.assert_allclose(gk, gk_ref)
        np.testing.assert_allclose(gb, gb_ref)


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


# ---------------------------------------------------------------------------
# Patch-extraction ground truth (conv == einsum over extracted patches)
# ---------------------------------------------------------------------------

class TestConvPatchExtraction:
    """Pin the ``conv_general_dilated_patches`` channel-ordering convention.

    The per-position D-RTRL kernel trace relies on
    ``_conv_extract_patches`` returning patches laid out as
    ``(batch, *spatial_out, c_in, *kernel_spatial)`` such that

    .. math::  y_{b,\\mathbf{s},k} = \\sum_{\\mathbf{u},c}
               \\mathrm{patch}_{b,\\mathbf{s},c,\\mathbf{u}}\\, K_{...}

    reproduces ``jax.lax.conv_general_dilated`` exactly. This is verified
    *numerically* (not trusted from the docs) for the default NCH/OIH
    layout, a strided conv, VALID padding and a channel-last layout.
    """

    def _patches(self, x, kernel_shape, **params):
        from braintrace._op.conv import _conv_extract_patches
        return _conv_extract_patches(x, kernel_shape, params)

    def test_patches_match_conv_default_layout(self):
        brainstate.random.seed(0)
        x = brainstate.random.randn(2, 3, 8)        # NCH
        k = brainstate.random.randn(4, 3, 3)        # OIH
        params = dict(strides=(1,), padding='SAME', dimension_numbers=None)
        patches = self._patches(x, k.shape, **params)
        assert patches.shape == (2, 8, 3, 3)        # (b, s, c_in, u)
        y = jnp.einsum('bscu,kcu->bks', patches, k)  # back to NCH
        ref = jax.lax.conv_general_dilated(x, k, window_strides=(1,), padding='SAME')
        np.testing.assert_allclose(y, ref, atol=1e-5)

    def test_patches_match_conv_strided(self):
        brainstate.random.seed(1)
        x = brainstate.random.randn(2, 3, 8)
        k = brainstate.random.randn(4, 3, 3)
        params = dict(strides=(2,), padding='SAME', dimension_numbers=None)
        patches = self._patches(x, k.shape, **params)
        y = jnp.einsum('bscu,kcu->bks', patches, k)
        ref = jax.lax.conv_general_dilated(x, k, window_strides=(2,), padding='SAME')
        np.testing.assert_allclose(y, ref, atol=1e-5)

    def test_patches_match_conv_valid_padding(self):
        brainstate.random.seed(2)
        x = brainstate.random.randn(2, 3, 8)
        k = brainstate.random.randn(4, 3, 3)
        params = dict(strides=(1,), padding='VALID', dimension_numbers=None)
        patches = self._patches(x, k.shape, **params)
        assert patches.shape == (2, 6, 3, 3)        # L_out = 8 - 3 + 1
        y = jnp.einsum('bscu,kcu->bks', patches, k)
        ref = jax.lax.conv_general_dilated(x, k, window_strides=(1,), padding='VALID')
        np.testing.assert_allclose(y, ref, atol=1e-5)

    def test_patches_match_conv_channel_last(self):
        brainstate.random.seed(3)
        x = brainstate.random.randn(2, 8, 3)        # NWC
        k = brainstate.random.randn(3, 3, 4)        # WIO
        dn = ('NWC', 'WIO', 'NWC')
        params = dict(strides=(1,), padding='SAME', dimension_numbers=dn)
        patches = self._patches(x, k.shape, **params)
        assert patches.shape == (2, 8, 3, 3)        # (b, s, c_in, u)
        y = jnp.einsum('bscu,uco->bso', patches, k)  # back to NWC
        ref = jax.lax.conv_general_dilated(
            x, k, window_strides=(1,), padding='SAME', dimension_numbers=dn
        )
        np.testing.assert_allclose(y, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Exact-gradient oracle: param-dim D-RTRL vs BPTT (per-position kernel trace)
# ---------------------------------------------------------------------------

class TestConvOnlineLearningExact:
    """Single-step D-RTRL must reproduce BPTT exactly for conv kernel + bias.

    The models are exactly diagonal (leaky-integrator dynamics
    ``h <- leak * h + drive``), so D-RTRL's diagonal approximation is exact
    and every parameter gradient must match a BPTT oracle element-wise
    (``rel < 1e-10`` in float64). This covers the audit finding that the
    kernel-shaped (spatially pre-summed) trace multiplied a sum by a sum
    where a sum of products is required — kernel gradients were wrong even
    at T=1 (measured rel err 1.0–47.8) and the bias recurrence used the
    spatially-summed multiplier (exact at T=1, rel err ~6 at T=3).
    """

    LEAK = 0.5
    TOL = 1e-10

    @staticmethod
    def _rel_err(a, b):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        denom = jnp.maximum(jnp.abs(a).max(), 1e-12)
        return float(jnp.abs(a - b).max() / denom)

    def _assert_exact(self, factory, xs):
        from braintrace._algorithm.oracle import (
            bptt_param_gradients,
            online_param_gradients_singlestep_naive,
        )
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = self._rel_err(g_bptt[key], g_online[key])
            assert rel < self.TOL, (
                f'D-RTRL diverges from BPTT for {key} at T={xs.shape[0]}: '
                f'max_rel_err={rel:.3e}'
            )

    def test_default_layout_kernel_exact(self):
        """NCH/OIH default layout: kernel gradient exact at T=1 and T=4."""
        c_in, c_out, length = 2, 3, 8
        leak = self.LEAK
        with brainstate.environ.context(precision=64):
            brainstate.random.seed(0)
            k0 = 0.1 * brainstate.random.randn(c_out, c_in, 3)  # OIH

            def factory():
                class Net(brainstate.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.k = brainstate.ParamState(k0)
                        self.h = brainstate.HiddenState(jnp.zeros((1, c_out, length)))  # NCH

                    def update(self, x):
                        drive = braintrace.conv(
                            x, self.k.value, strides=(1,), padding='SAME'
                        )
                        self.h.value = leak * self.h.value + drive
                        return self.h.value

                return Net()

            for T in (1, 4):
                brainstate.random.seed(42)
                xs = 0.3 * brainstate.random.randn(T, 1, c_in, length)
                self._assert_exact(factory, xs)

    def test_channel_last_with_bias_exact(self):
        """NWC/WIO/NWC layout with a bias: kernel *and* bias exact at T=3.

        Pins the bias-recurrence fix — with the old spatially-summed
        recurrence multiplier the bias gradient failed at T >= 2 (rel err ~6).
        """
        c_in, c_out, width = 2, 3, 8
        leak = self.LEAK
        dn = ('NWC', 'WIO', 'NWC')
        with brainstate.environ.context(precision=64):
            brainstate.random.seed(1)
            k0 = 0.1 * brainstate.random.randn(3, c_in, c_out)  # WIO
            b0 = 0.05 * brainstate.random.randn(c_out)

            def factory():
                class Net(brainstate.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.k = brainstate.ParamState(k0)
                        self.b = brainstate.ParamState(b0)
                        self.h = brainstate.HiddenState(jnp.zeros((1, width, c_out)))  # NWC

                    def update(self, x):
                        drive = braintrace.conv(
                            x, self.k.value, self.b.value,
                            strides=(1,), padding='SAME', dimension_numbers=dn,
                        )
                        self.h.value = leak * self.h.value + drive
                        return self.h.value

                return Net()

            brainstate.random.seed(43)
            xs = 0.3 * brainstate.random.randn(3, 1, width, c_in)
            self._assert_exact(factory, xs)


# ---------------------------------------------------------------------------
# Grouped-conv rejection under param-dim D-RTRL
# ---------------------------------------------------------------------------

class TestGroupedConvDrtrlRejection:
    """``feature_group_count != 1`` has no per-position patch extraction under
    the exact param-dim D-RTRL kernel trace; trace init must reject it with a
    pointer to the IO-dim alternative (``pp_prop``)."""

    def test_feature_group_count_rejected_with_pp_prop_pointer(self):
        c_in, c_out, length = 4, 4, 8
        with brainstate.environ.context(precision=64):
            brainstate.random.seed(0)
            k0 = 0.1 * brainstate.random.randn(c_out, c_in // 2, 3)  # OIH, groups=2

            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.k = brainstate.ParamState(k0)
                    self.h = brainstate.HiddenState(jnp.zeros((1, c_out, length)))

                def update(self, x):
                    drive = braintrace.conv(
                        x, self.k.value, strides=(1,), padding='SAME',
                        feature_group_count=2,
                    )
                    self.h.value = 0.5 * self.h.value + drive
                    return self.h.value

            net = Net()
            brainstate.nn.init_all_states(net, batch_size=1)
            algo = braintrace.D_RTRL(net, vjp_method='single-step')
            # ``compile_graph`` initialises the trace states, which is where
            # the per-position trace allocation rejects grouped convs.
            with pytest.raises(NotImplementedError, match='pp_prop'):
                algo.compile_graph(jnp.zeros((1, c_in, length)))


# ---------------------------------------------------------------------------
# Audit Task 11 (T3): first-principles ``instant_drtrl`` / ``solve_drtrl``
# from ``jax.jacobian``
# ---------------------------------------------------------------------------

class TestInstantSolveDrtrlFirstPrinciplesFromJacobian:
    """Derive ``_conv_instant_drtrl`` / ``_conv_solve_drtrl`` from
    ``jax.jacobian`` of the primitive's own forward, on tiny shapes, default
    (OIH kernel / NCH data) layout, unbatched.

    ``instant_drtrl``'s documented formula is
    ``(dh/dK)_{s,u,c,k} = D_f^t_{s,k} * patch_{s,u,c}``, i.e. the per-position
    (spatial index ``s`` never summed) contraction of the cotangent against
    the raw conv Jacobian ``dy_{k,s}/dK_{k2,c,u} = delta(k,k2) * patch_{s,u,c}``.
    This test builds the full Jacobian via ``jax.jacobian`` (never via the
    rule's own patch-extraction helper), verifies its diagonal-in-out-channel
    structure against an independently hand-built "SAME"-padding patch
    (plain zero-padding + indexing, not :func:`_conv_extract_patches`), then
    contracts the raw Jacobian with a random cotangent via a *repeated*-index
    ``einsum`` (the spatial axis ``s`` appears in both operands and the
    output, so it is kept rather than summed) and compares against the
    rule's actual output.

    ``solve_drtrl`` is checked separately: it performs a plain spatial-sum
    contraction between a random loss cotangent and a random trace (no
    forward model to differentiate), reimplemented here independently of
    the rule's own axis bookkeeping.
    """

    def test_instant_drtrl_weight_matches_jacobian_repeated_index_contraction(self):
        brainstate.random.seed(701)
        in_ch, out_ch, kw, L = 2, 4, 3, 6  # odd kernel width: symmetric SAME pad
        x = brainstate.random.randn(in_ch, L)  # unbatched, default NCH-style
        K0 = brainstate.random.randn(out_ch, in_ch, kw)  # default OIH kernel

        def fwd(K):
            y = jax.lax.conv_general_dilated(
                x[None], K, window_strides=(1,), padding='SAME',
            )
            return y[0]  # (out_ch, L)

        J = jax.jacobian(fwd)(K0)  # (k, s, k2, c, u)

        pad = (kw - 1) // 2
        x_padded = jnp.pad(x, ((0, 0), (pad, pad)))

        def manual_patch(s, u, c):
            return x_padded[c, s + u]

        for k in range(out_ch):
            for s in range(L):
                for k2 in range(out_ch):
                    for c in range(in_ch):
                        for u in range(kw):
                            expected = float(manual_patch(s, u, c)) if k == k2 else 0.0
                            np.testing.assert_allclose(
                                J[k, s, k2, c, u], expected, atol=1e-6,
                                err_msg=f'Jacobian mismatch at k={k},s={s},k2={k2},c={c},u={u}',
                            )

        g = brainstate.random.randn(out_ch, L)  # hidden_dim, (k, s) layout

        # Repeated-index einsum: 's' appears in both operands and the output
        # so it is NOT summed -- only the y-channel index (labelled 'k' on
        # the cotangent, 'm' on the Jacobian's differentiation side) is
        # contracted, matching the plan's "g-contraction ... per position".
        ref_inst = jnp.einsum('ks,ksmcu->smcu', g, J)

        from braintrace._op.conv import _conv_instant_drtrl
        out = _conv_instant_drtrl(
            x, g, {'weight': K0}, strides=(1,), padding='SAME', has_bias=False,
        )
        np.testing.assert_allclose(out['weight'], ref_inst, atol=1e-6)

    def test_instant_drtrl_matches_jacobian_with_kernel_fn_and_bias_fn(self):
        brainstate.random.seed(702)
        in_ch, out_ch, kw, L = 2, 4, 3, 6
        x = brainstate.random.randn(in_ch, L)
        K0 = brainstate.random.randn(out_ch, in_ch, kw)
        b0 = brainstate.random.randn(out_ch)

        kernel_fn = lambda K: 1.5 * K + 0.1 * K ** 2
        bias_fn = lambda b: jnp.tanh(b)

        def fwd(K):
            y = jax.lax.conv_general_dilated(
                x[None], kernel_fn(K), window_strides=(1,), padding='SAME',
            )
            return y[0]

        J = jax.jacobian(fwd)(K0)  # w.r.t. the RAW kernel; kernel_fn is inside fwd
        g = brainstate.random.randn(out_ch, L)
        ref_inst = jnp.einsum('ks,ksmcu->smcu', g, J)

        from braintrace._op.conv import _conv_instant_drtrl
        out = _conv_instant_drtrl(
            x, g, {'weight': K0}, strides=(1,), padding='SAME', has_bias=False,
            kernel_fn=kernel_fn,
        )
        np.testing.assert_allclose(out['weight'], ref_inst, atol=1e-5)

        # Bias: jacobian of bias_fn itself (diagonal), never assumed.
        J_b = jax.jacobian(bias_fn)(b0)
        db = jnp.diagonal(J_b)
        ref_bias = g * db[:, None]  # channel axis is axis 0 in (k, s) layout

        out_b = _conv_instant_drtrl(
            x, g, {'weight': K0, 'bias': b0}, strides=(1,), padding='SAME',
            has_bias=True, bias_fn=bias_fn,
        )
        np.testing.assert_allclose(out_b['bias'], ref_bias, atol=1e-6)

    def test_solve_drtrl_matches_independent_spatial_sum_reference(self):
        from braintrace._op.conv import _conv_solve_drtrl
        brainstate.random.seed(703)
        in_ch, out_ch, kw, L = 2, 4, 3, 6
        K0 = brainstate.random.randn(out_ch, in_ch, kw)

        trace_w = brainstate.random.randn(L, out_ch, in_ch, kw)  # (*spatial_out, *kernel)
        trace_b = brainstate.random.randn(out_ch, L)  # (*y_layout)
        dg_hidden = brainstate.random.randn(out_ch, L)  # (*y_layout)

        # Independent reimplementation of the documented spatial-sum
        # contraction, built without reusing `_conv_layout` or any of the
        # rule's own axis-alignment code.
        dg_t = jnp.transpose(dg_hidden, (1, 0))  # (s, k)
        ref_w = jnp.einsum('sk,skcu->kcu', dg_t, trace_w)
        ref_b = jnp.sum(trace_b * dg_hidden, axis=1)

        out = _conv_solve_drtrl(
            dg_hidden, {'weight': trace_w, 'bias': trace_b}, {'weight': K0},
            strides=(1,), padding='SAME', has_bias=True,
        )
        np.testing.assert_allclose(out['weight'], ref_w, atol=1e-6)
        np.testing.assert_allclose(out['bias'], ref_b, atol=1e-6)
