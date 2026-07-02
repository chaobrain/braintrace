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

from __future__ import annotations

import brainstate
import brainunit as u
import jax.numpy as jnp

from braintrace._op import embedding
from braintrace._typing import ArrayLike

__all__ = ['Embedding']


class Embedding(brainstate.nn.Embedding):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Embedding.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, indices: ArrayLike) -> ArrayLike:
        """Look up embeddings through the ETP ``embedding`` primitive.

        Routing the gather through :func:`braintrace.embedding` is what makes
        the table eligible for online-learning trace computation. Indices of
        rank 2 or higher are folded into one flat axis before the op (the
        rank-guarded primitive accepts only scalar or ``(batch,)`` indices)
        and the output is unfolded to ``(*indices.shape, features)``.

        ``max_norm``, ``freeze``, ``scale_grad_by_freq``, and ``padding_idx``
        modify the lookup or its gradient outside the primitive that online
        learning traces, so they raise ``NotImplementedError`` instead of
        silently diverging from the ``brainstate`` semantics.

        Parameters
        ----------
        indices : ArrayLike
            Integer token indices of any rank.

        Returns
        -------
        ArrayLike
            The gathered embeddings, of shape ``(*indices.shape, features)``.

        Raises
        ------
        NotImplementedError
            If the layer was constructed with ``max_norm``, ``freeze``,
            ``scale_grad_by_freq``, or ``padding_idx``.
        """
        if (
            self.max_norm is not None
            or self.freeze
            or self.scale_grad_by_freq
            or self.padding_idx is not None
        ):
            raise NotImplementedError(
                'braintrace.nn.Embedding does not support max_norm, freeze, '
                'scale_grad_by_freq, or padding_idx: they modify the lookup '
                'or its gradient outside the ETP primitive that online '
                'learning requires.'
            )
        indices = jnp.asarray(indices)
        table = self.weight.value
        if indices.ndim <= 1:
            return embedding(indices, table)
        # fold all index axes into one batch axis, unfold on the output
        # (reshape via brainunit so quantities keep their units)
        y = embedding(indices.reshape(-1), table)
        return u.math.reshape(y, (*indices.shape, y.shape[-1]))  # type: ignore[union-attr]
