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

"""Tests for braintrace.nn.Embedding."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


class TestEmbedding:

    def test_lookup_matches_table(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([0, 3, 3], dtype=jnp.int32)
        np.testing.assert_allclose(layer(idx), layer.weight.value[idx], atol=1e-6)

    def test_scalar_lookup(self):
        layer = braintrace.nn.Embedding(10, 4)
        out = layer(jnp.int32(7))
        assert out.shape == (4,)
        np.testing.assert_allclose(out, layer.weight.value[7], atol=1e-6)

    def test_folds_extra_leading_axes(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        out = layer(idx)
        assert out.shape == (2, 3, 4)
        np.testing.assert_allclose(out, layer.weight.value[idx], atol=1e-6)

    def test_uses_etp_primitive(self):
        layer = braintrace.nn.Embedding(10, 4)
        jp = jax.make_jaxpr(lambda i: layer(i))(jnp.array([0, 1], dtype=jnp.int32))
        prims = {str(e.primitive) for e in jp.jaxpr.eqns}
        assert 'etp_emb' in prims
        assert 'gather' not in prims

    def test_gradient_flows_to_table(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([2, 2], dtype=jnp.int32)
        g = brainstate.transform.grad(
            lambda: layer(idx).sum(), layer.states(brainstate.ParamState))()
        (gval,) = g.values()
        want = jnp.zeros((10, 4)).at[jnp.array([2, 2])].add(1.0)
        np.testing.assert_allclose(gval, want, atol=1e-6)

    def test_exported_from_nn(self):
        assert 'Embedding' in braintrace.nn.__all__


class TestEmbeddingUnsupportedOptions:
    """E-04: the four unsupported options are rejected by ``__init__``.

    Before this fix the constructor accepted all four and ``update()`` raised,
    so the failure surfaced at the first forward pass -- under ``jit``,
    arbitrarily far from the line that passed the option.
    """

    @pytest.mark.parametrize(
        'kwargs, named',
        [
            ({'max_norm': 1.0}, 'max_norm'),
            ({'freeze': True}, 'freeze'),
            ({'scale_grad_by_freq': True}, 'scale_grad_by_freq'),
            ({'padding_idx': 0}, 'padding_idx'),
        ],
    )
    def test_each_option_raises_at_construction(self, kwargs, named):
        with pytest.raises(NotImplementedError) as exc:
            braintrace.nn.Embedding(10, 4, **kwargs)
        assert named in str(exc.value)

    def test_message_names_every_offender_and_no_others(self):
        with pytest.raises(NotImplementedError) as exc:
            braintrace.nn.Embedding(
                10, 4, freeze=True, padding_idx=2, scale_grad_by_freq=True)
        # The offenders appear with their values, in the fixed declaration
        # order; the option that was left at its default is not reported.
        msg = str(exc.value)
        offenders = msg.split('does not support: ', 1)[1].split('.', 1)[0]
        assert offenders == 'freeze=True, scale_grad_by_freq=True, padding_idx=2'

    def test_failure_is_at_init_not_at_first_call(self):
        """The regression: no forward pass is needed to provoke the error.

        The old behaviour was for this constructor to return an object and for
        the error to appear only when that object was called. If the layer is
        ever constructed here, ``pytest.raises`` fails and the regression is
        caught.
        """
        with pytest.raises(NotImplementedError):
            braintrace.nn.Embedding(10, 4, max_norm=1.0)

    def test_jitted_forward_is_never_reached(self):
        """Under ``jit`` the old failure fired at trace time, not construction.

        Construct inside the jitted callable's *builder*, so that if the layer
        constructed successfully the error could only come from the traced
        call. Asserting the callable was never invoked pins the timing.
        """
        called = []

        def build_and_run(idx):
            layer = braintrace.nn.Embedding(10, 4, scale_grad_by_freq=True)
            called.append(True)
            return layer(idx)

        with pytest.raises(NotImplementedError):
            brainstate.transform.jit(build_and_run)(jnp.array([0, 1], dtype=jnp.int32))
        assert called == [], 'construction should have raised before the lookup'

    def test_explicit_defaults_construct_and_run(self):
        layer = braintrace.nn.Embedding(
            10, 4,
            max_norm=None,
            freeze=False,
            scale_grad_by_freq=False,
            padding_idx=None,
        )
        idx = jnp.array([1, 5], dtype=jnp.int32)
        np.testing.assert_allclose(layer(idx), layer.weight.value[idx], atol=1e-6)

    def test_norm_type_is_not_in_the_unsupported_set(self):
        # `norm_type` only matters together with `max_norm`, which is rejected,
        # so it stays accepted and inert rather than joining the rejected set.
        layer = braintrace.nn.Embedding(10, 4, norm_type=1.0)
        assert layer(jnp.array([0], dtype=jnp.int32)).shape == (1, 4)

    def test_parent_value_error_still_wins_for_out_of_range_padding_idx(self):
        # An out-of-range index is a different mistake from an unsupported
        # feature; the parent's more specific diagnosis must not be masked.
        with pytest.raises(ValueError) as exc:
            braintrace.nn.Embedding(10, 4, padding_idx=99)
        assert not isinstance(exc.value, NotImplementedError)

    @pytest.mark.parametrize(
        'attr, value',
        [
            ('max_norm', 1.0),
            ('freeze', True),
            ('scale_grad_by_freq', True),
            ('padding_idx', 0),
        ],
    )
    def test_post_init_mutation_is_still_caught_by_update(self, attr, value):
        """``update()``'s check is not dead code.

        These are plain public attributes assigned by the parent constructor,
        so a caller can enable one after construction. Dropping the check in
        ``update()`` would make that silently produce the wrong semantics.
        """
        layer = braintrace.nn.Embedding(10, 4)
        setattr(layer, attr, value)
        with pytest.raises(NotImplementedError) as exc:
            layer(jnp.array([0], dtype=jnp.int32))
        assert attr in str(exc.value)

    def test_from_pretrained_rejects_its_default_freeze(self):
        # `from_pretrained` defaults to `freeze=True`, so it now fails at the
        # `from_pretrained` call rather than at the first forward pass.
        w = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        with pytest.raises(NotImplementedError) as exc:
            braintrace.nn.Embedding.from_pretrained(w)
        assert 'freeze' in str(exc.value)

    def test_from_pretrained_works_with_freeze_disabled(self):
        w = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        layer = braintrace.nn.Embedding.from_pretrained(w, freeze=False)
        out = layer(jnp.array([1], dtype=jnp.int32))
        np.testing.assert_allclose(out, w[jnp.array([1])], atol=1e-6)
