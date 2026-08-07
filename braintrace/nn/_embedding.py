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

from typing import Callable, Optional, Union

import brainstate
import brainunit as u
import jax.numpy as jnp

from braintrace._op import embedding
from braintrace._typing import ArrayLike, Size

__all__ = ['Embedding']


def _reject_unsupported_options(
    max_norm: Optional[float],
    freeze: bool,
    scale_grad_by_freq: bool,
    padding_idx: Optional[int],
) -> None:
    """Raise if any option outside the ETP ``embedding`` primitive is enabled.

    ``max_norm`` and ``freeze`` wrap the lookup in ``stop_gradient``;
    ``scale_grad_by_freq`` and ``padding_idx`` live in the ``brainstate``
    parent's ``custom_vjp`` backward rule, which the ETP ``embedding``
    primitive replaces outright. None of the four is visible to the
    online-learning trace machinery, so honouring them silently would diverge
    from the ``brainstate`` semantics rather than reproduce them.

    Shared by :meth:`Embedding.__init__` and :meth:`Embedding.update` so that
    the unsupported set and its message cannot drift apart. The constructor is
    the gate that matters; ``update`` re-checks because these are plain public
    attributes a caller can still assign after construction.

    Parameters
    ----------
    max_norm : float or None
        The ``max_norm`` option. Unsupported unless ``None``.
    freeze : bool
        The ``freeze`` option. Unsupported unless ``False``.
    scale_grad_by_freq : bool
        The ``scale_grad_by_freq`` option. Unsupported unless ``False``.
    padding_idx : int or None
        The ``padding_idx`` option. Unsupported unless ``None``.

    Raises
    ------
    NotImplementedError
        If any argument is set to a value other than its default. The message
        names every offending option and its value, in the fixed order
        ``max_norm``, ``freeze``, ``scale_grad_by_freq``, ``padding_idx``.
    """
    offenders = []
    if max_norm is not None:
        offenders.append(f'max_norm={max_norm!r}')
    if freeze:
        offenders.append(f'freeze={freeze!r}')
    if scale_grad_by_freq:
        offenders.append(f'scale_grad_by_freq={scale_grad_by_freq!r}')
    if padding_idx is not None:
        offenders.append(f'padding_idx={padding_idx!r}')
    if not offenders:
        return
    raise NotImplementedError(
        f'braintrace.nn.Embedding does not support: {", ".join(offenders)}. '
        'These modify the lookup or its gradient outside the ETP primitive '
        'that online learning traces. Use braintrace.nn.Embedding with '
        'default values for max_norm, freeze, scale_grad_by_freq and '
        'padding_idx, or use brainstate.nn.Embedding if you do not need '
        'online-learning traces.'
    )


class Embedding(brainstate.nn.Embedding):
    r"""A lookup table whose gather is routed through the ETP ``embedding`` op.

    Drop-in replacement for :class:`brainstate.nn.Embedding` that performs the
    table lookup with :func:`braintrace.embedding`, which is what makes the
    table eligible for online-learning trace computation. Indices of any rank
    are accepted; rank 2 and above are folded into one flat axis before the op
    (the rank-guarded primitive takes only scalar or ``(batch,)`` indices) and
    the output is unfolded back to ``(*indices.shape, *embedding_size)``.

    Four of the parent's options are **accepted for signature compatibility but
    not supported, and are rejected by the constructor**: ``max_norm``,
    ``freeze``, ``scale_grad_by_freq`` and ``padding_idx``. Each modifies the
    lookup or its gradient outside the ETP primitive that online learning
    traces, so there is no way to honour them without diverging from the
    ``brainstate`` semantics. They are part of the signature so that code
    written against :class:`brainstate.nn.Embedding` fails with a clear message
    at the constructor call rather than with a ``TypeError`` about an unexpected
    keyword — or, as before this validation moved, with a deferred failure at
    the first forward pass, which under ``jit`` can be far from the mistake.

    Parameters
    ----------
    num_embeddings : int
        Size of the embedding dictionary. Must be non-negative.
    embedding_size : int or sequence of int
        Size of each embedding vector.
    embedding_init : Callable or ArrayLike, optional
        Initializer for the lookup table, of shape
        ``(num_embeddings, *embedding_size)``. Default is ``LecunUniform()``.
    padding_idx : int, optional
        Accepted but **not supported**; anything other than ``None`` raises
        ``NotImplementedError`` from the constructor. Zeroing the gradient of
        one row happens in the parent's backward rule, which the ETP primitive
        replaces.
    max_norm : float, optional
        Accepted but **not supported**; anything other than ``None`` raises
        ``NotImplementedError`` from the constructor. Renormalizing rows
        inserts a ``stop_gradient`` the trace machinery cannot see through.
    norm_type : float, optional
        The p of the p-norm used by ``max_norm``. Supported in the sense that
        it is accepted and stored, but inert: it only has an effect together
        with ``max_norm``, which is rejected. Default is ``2.0``.
    scale_grad_by_freq : bool, optional
        Accepted but **not supported**; ``True`` raises
        ``NotImplementedError`` from the constructor. The inverse-frequency
        scaling lives in the parent's backward rule, which the ETP primitive
        replaces. Default is ``False``.
    freeze : bool, optional
        Accepted but **not supported**; ``True`` raises
        ``NotImplementedError`` from the constructor. Freezing wraps the table
        in ``stop_gradient``, which removes the very gradient path online
        learning traces. Default is ``False``.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Parameter state type. Default is ``brainstate.ParamState``.

    Attributes
    ----------
    weight : brainstate.ParamState
        The learnable table, of shape ``(num_embeddings, *embedding_size)``.

    Raises
    ------
    NotImplementedError
        From the constructor, if ``max_norm``, ``freeze``,
        ``scale_grad_by_freq`` or ``padding_idx`` is set to a non-default
        value. The message names every offending option.

    See Also
    --------
    braintrace.embedding : The ETP primitive this layer wraps.
    brainstate.nn.Embedding : The upstream layer, which supports all options.

    Examples
    --------
    Look up rows of the table:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.Embedding(10, 4)
        >>> layer(jnp.array([0, 3, 3])).shape
        (3, 4)

    Index arrays of rank 2 or higher are folded and unfolded automatically:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.Embedding(10, 4)
        >>> layer(jnp.array([[0, 1, 2], [3, 4, 5]])).shape
        (2, 3, 4)

    An unsupported option fails at the constructor, not at the forward pass:

    .. code-block:: python

        >>> import braintrace
        >>> braintrace.nn.Embedding(10, 4, freeze=True)
        Traceback (most recent call last):
            ...
        NotImplementedError: braintrace.nn.Embedding does not support: freeze=True. ...
    """

    __module__ = 'braintrace.nn'

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: Size,
        embedding_init: Union[Callable, ArrayLike] = brainstate.nn.init.LecunUniform(),
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        freeze: bool = False,
        name: Optional[str] = None,
        param_type: type = brainstate.ParamState,
    ):
        """Build the table and reject the options this layer cannot trace.

        The parent constructor runs first so that its own argument validation
        keeps producing the more specific diagnosis where it has one — an
        out-of-range ``padding_idx`` is still the parent's ``ValueError``, not
        an "unsupported option" report, because being out of range is a
        different mistake.

        Every argument is forwarded to :class:`brainstate.nn.Embedding`
        unchanged and is documented on the class docstring above.

        Raises
        ------
        NotImplementedError
            If ``max_norm``, ``freeze``, ``scale_grad_by_freq`` or
            ``padding_idx`` is set to a non-default value.
        """
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            embedding_init=embedding_init,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            freeze=freeze,
            name=name,
            param_type=param_type,
        )
        _reject_unsupported_options(
            self.max_norm, self.freeze, self.scale_grad_by_freq, self.padding_idx
        )

    def update(self, indices: ArrayLike) -> ArrayLike:
        """Look up embeddings through the ETP ``embedding`` primitive.

        Routing the gather through :func:`braintrace.embedding` is what makes
        the table eligible for online-learning trace computation. Indices of
        rank 2 or higher are folded into one flat axis before the op (the
        rank-guarded primitive accepts only scalar or ``(batch,)`` indices)
        and the output is unfolded to ``(*indices.shape, features)``.

        The unsupported-option check is re-run here, not only in
        ``__init__``. ``max_norm``, ``freeze``, ``scale_grad_by_freq`` and
        ``padding_idx`` are plain public attributes that the parent
        constructor assigns directly, so a caller can still enable one after
        construction (``layer.freeze = True``); without this second gate that
        would silently produce the wrong semantics instead of an error.

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
            If ``max_norm``, ``freeze``, ``scale_grad_by_freq`` or
            ``padding_idx`` was enabled by assignment after construction.
            Passing one to the constructor raises there instead.
        """
        _reject_unsupported_options(
            self.max_norm, self.freeze, self.scale_grad_by_freq, self.padding_idx
        )
        indices = jnp.asarray(indices)
        table = self.weight.value
        if indices.ndim <= 1:
            return embedding(indices, table)
        # fold all index axes into one batch axis, unfold on the output
        # (reshape via brainunit so quantities keep their units)
        y = embedding(indices.reshape(-1), table)
        return u.math.reshape(y, (*indices.shape, y.shape[-1]))  # type: ignore[union-attr]
