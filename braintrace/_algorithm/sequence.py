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

"""Sequence drivers: run a compiled learner over a sequence.

Spec: ``docs/specs/2026-07-27-sequence-driver-api.md``.

This is a *composition* layer. It owns the loop, the gradient accumulator and
the loss reduction; it owns no numerics. The model call stays in the caller's
``step_fn``, because that call is where the eligibility trace advances and
hiding it would hide the subject of the library.

Two encodings are load-bearing and easy to get wrong silently:

- ``chunk_size=1`` is the **plain** path, matching ``dni._as_window``. Window
  mode starts at ``k >= 2``. A user who fits a synthesiser at
  ``train_synthetic_gradient(chunk_size=1)`` and drives at
  ``etrace_grad(chunk_size=1)`` must be right to believe they matched (F-35).
- ``mask`` gates the **loss only**. The learner is driven at every step, so a
  zero-weighted step still shapes the trace that later weighted steps consume.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import brainstate
import jax
import jax.numpy as jnp

from .._input_data import MultiStepData, is_input

__all__ = [
    'SequenceDriverMixin',
    'ETraceVmap',
]

_REDUCTIONS = ('mean', 'sum')
_LOSS_OUTPUTS = ('per_step', 'masked', 'scalar')


def _check_chunk_size(chunk_size: Any) -> Optional[int]:
    """Validate ``chunk_size`` and normalize it to ``None`` or ``int >= 2``.

    ``1`` folds to ``None``: the two are synonyms for the plain path, matching
    ``dni._as_window``'s ``chunk_size == 1`` case.
    """
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(
            f'chunk_size must be None or an int, got {type(chunk_size).__name__} '
            f'({chunk_size!r}). Pass a Python int, not a traced or numpy value.'
        )
    if chunk_size < 1:
        raise ValueError(f'chunk_size must be at least 1, got {chunk_size}.')
    # 1 is the plain single-step path, not a length-1 window -- the same
    # encoding `train_synthetic_gradient` uses.
    return None if chunk_size == 1 else chunk_size


def _sequence_length(sequences: tuple) -> int:
    """The common leading length of every leaf, or a ``ValueError`` naming the leaf."""
    if len(sequences) == 0:
        raise ValueError(
            'etrace_grad/etrace_evolve needs at least one sequence: the '
            'sequences define the length T to iterate over. Pass the '
            'time-major arrays you want sliced, e.g. '
            'learner.etrace_grad(inputs, targets, step_fn=...).'
        )

    for pos, seq in enumerate(sequences):
        leaves = jax.tree.leaves(seq, is_leaf=is_input)
        if any(is_input(leaf) for leaf in leaves):
            raise TypeError(
                f'sequence {pos} is (or contains) a SingleStepData / '
                f'MultiStepData wrapper. Those describe how one *slice* reaches '
                f'the model, which is step_fn\'s decision -- and the driver '
                f'slices along axis 0, which would decompose the wrapper rather '
                f'than the data. Pass the raw array here and wrap inside step_fn.'
            )

    length: Optional[int] = None
    origin = None
    for pos, seq in enumerate(sequences):
        leaves_with_path = jax.tree.leaves_with_path(seq)
        if not leaves_with_path:
            raise ValueError(
                f'sequence {pos} is an empty pytree ({seq!r}), so it holds no '
                f'array to take a leading length from. Pass the time-major '
                f'arrays you want sliced, or drop the argument entirely.'
            )
        for path, leaf in leaves_with_path:
            shape = jnp.shape(leaf)
            if len(shape) == 0:
                raise ValueError(
                    f'sequence {pos}{jax.tree_util.keystr(path)} is a scalar, so '
                    f'it has no leading time axis to slice. Constant arguments '
                    f'should be closed over by step_fn instead of passed as '
                    f'sequences.'
                )
            if length is None:
                length, origin = shape[0], f'sequence {pos}{jax.tree_util.keystr(path)}'
            elif shape[0] != length:
                raise ValueError(
                    f'sequences must share a leading length: {origin} has length '
                    f'{length}, but sequence {pos}{jax.tree_util.keystr(path)} has '
                    f'length {shape[0]}.'
                )

    assert length is not None
    if length == 0:
        raise ValueError(
            'the sequences are empty (leading length 0), so there is no '
            'objective to reduce and no step to drive.'
        )
    return length


def _check_mask(mask: Any, length: int):
    """Validate ``mask`` and return it as a ``(T,)`` array of weights."""
    if mask is None:
        return jnp.ones((length,), dtype=jnp.float32)
    mask = jnp.asarray(mask)
    if mask.shape != (length,):
        raise ValueError(
            f'mask must have shape (T,) = ({length},) to weight one loss per '
            f'step, got {mask.shape}.'
        )
    return mask


def _to_windows(tree, n_windows: int, chunk_size: int):
    """Reshape every leaf ``(T, ...) -> (n_windows, chunk_size, ...)``."""
    return jax.tree.map(
        lambda a: a.reshape((n_windows, chunk_size) + jnp.shape(a)[1:]), tree
    )


class SequenceDriverMixin:
    """``etrace_grad`` / ``etrace_evolve``, written once for both hosts.

    Hosts supply three hooks:

    ``_seq_call``
        The callable that drives one step or window. Defaults to ``self``.
    ``_seq_param_states``
        The default set of :class:`brainstate.ParamState` to differentiate.
    ``_seq_vjp_method``
        The learner's ``vjp_method``, or ``None`` if it has none.

    ``_seq_vjp_method`` is a hook rather than a ``getattr`` on the driver
    because ``brainstate.nn.Vmap`` defines no ``__getattr__`` and so does
    not forward ``vjp_method`` from ``.module``. Reading the attribute off the
    driver object would silently yield ``None`` for every vmapped learner and
    bypass the window-mode validation entirely.
    """

    #: Set by :class:`ETraceVmap`. Window mode is refused when true, because
    #: ``compile(vmap=True)`` maps ``in_axes=0`` and a ``(k, B, ...)`` window
    #: slice would map *time* as the batch axis.
    _seq_is_vmapped: bool = False

    @property
    def _seq_call(self) -> Callable:
        return self  # type: ignore[return-value]

    @property
    def _seq_param_states(self):
        raise NotImplementedError

    @property
    def _seq_vjp_method(self) -> Optional[str]:
        raise NotImplementedError

    # -- validation ---------------------------------------------------------

    def _seq_check_window(self, chunk_size: Optional[int], length: int,
                          *, for_grad: bool) -> None:
        """Gate window mode. ``chunk_size`` is already normalized (``None`` or ``>= 2``)."""
        if chunk_size is None:
            return

        if self._seq_is_vmapped:
            raise ValueError(
                f'chunk_size={chunk_size} (window mode) is not supported under '
                f'a vmapped learner. compile(vmap=True) maps in_axes=0, so a '
                f'(chunk_size, batch, ...) window slice would map the *time* '
                f'axis as the batch axis -- which is silently wrong whenever '
                f'chunk_size equals the batch size. Use the batched (non-vmap) '
                f'mode, which carries the batch axis inside the compiled graph, '
                f'or drive with chunk_size=None.'
            )

        if for_grad and self._seq_vjp_method != 'multi-step':
            raise ValueError(
                f'chunk_size={chunk_size} (window mode) needs a learner built '
                f'with vjp_method="multi-step", but this learner reports '
                f'{self._seq_vjp_method!r}. Rebuild it with '
                f'compile(..., vjp_method="multi-step"), or drive with '
                f'chunk_size=None. (Note this restriction applies to '
                f'etrace_grad only: etrace_evolve runs no loss VJP and accepts '
                f'windows on either vjp_method.)'
            )

        if length % chunk_size:
            raise ValueError(
                f'the sequence length {length} is not a multiple of '
                f'chunk_size {chunk_size}, so the last window would be '
                f'{length % chunk_size} step(s) long instead of {chunk_size}. '
                f'Ragged windows are refused rather than silently truncated; '
                f'trim to {length - length % chunk_size} steps, or pick a '
                f'chunk_size that divides {length}.'
            )

    # -- the gradient driver ------------------------------------------------

    def etrace_grad(
        self,
        *sequences: Any,
        step_fn: Callable,
        mask: Any = None,
        chunk_size: Optional[int] = None,
        weights: Any = None,
        reduction: str = 'mean',
        loss_output: str = 'per_step',
        has_aux: bool = False,
        return_value: bool = False,
    ):
        r"""Accumulate online gradients over a sequence.

        Parameters
        ----------
        *sequences
            One or more pytrees whose leaves share a leading length ``T``,
            sliced in lockstep and passed to ``step_fn`` positionally. There is
            no distinguished ``targets`` argument. May not be a
            :class:`SingleStepData` / :class:`MultiStepData` wrapper -- wrap
            inside ``step_fn`` instead.
        step_fn : callable
            The user's step function, which **runs the model itself** and
            returns the loss. Keyword-only and required. It must call this
            learner exactly once per invocation: zero calls leave the trace
            un-advanced for that step, two advance it twice, and neither is
            detectable from the returned gradient.
        mask : array, optional
            ``(T,)`` per-step loss weights; ``None`` means all-ones. Values need
            not be binary. Gates **only** the loss -- the model and the
            eligibility trace are driven at every step regardless.
        chunk_size : int, optional
            ``None`` (default) or ``1`` drive step-by-step, handing ``seq[t]``
            to ``step_fn``, which returns a scalar. ``k >= 2`` drives in
            windows, handing ``seq[t:t+k]`` of shape ``(k, ...)``, and
            ``step_fn`` must return a ``(k,)`` vector of per-step losses and
            wrap its model inputs in :class:`MultiStepData`. Window mode
            requires ``vjp_method='multi-step'`` and ``T % k == 0``, and is not
            available under a vmapped learner.
        weights : dict, optional
            The :class:`brainstate.ParamState` to differentiate. Defaults to
            the learner's own ``param_states``.
        reduction : {'mean', 'sum'}, optional
            ``'mean'`` (default) divides by the **total mask weight**,
            ``max(sum_t mask_t, 1)``, not by ``T``.
        loss_output : {'per_step', 'masked', 'scalar'}, optional
            What ``return_value=True`` hands back: the raw pre-mask losses
            ``(T,)``, the masked losses ``(T,)``, or the reduced objective
            (scalar). Ignored when ``return_value=False``.
        has_aux : bool, optional
            Whether ``step_fn`` returns ``(loss, aux)``.
        return_value : bool, optional
            Whether to return the losses alongside the gradients.

        Returns
        -------
        grads or tuple
            Mirrors ``brainstate.transform.grad``: ``grads``,
            ``(grads, losses)``, ``(grads, aux)`` or ``(grads, losses, aux)``.

        Raises
        ------
        TypeError
            If ``step_fn`` is not given; if *chunk_size* is not ``None`` or a
            Python ``int`` (a traced or numpy value is refused, since the value
            has to be known at trace time); or if a sequence is a
            :class:`SingleStepData` / :class:`MultiStepData` wrapper. The
            wrappers are registered pytree nodes, so slicing one would
            decompose the wrapper rather than the data.
        ValueError
            If no sequences are given, if their leading lengths disagree, or if
            ``T == 0`` -- there is nothing to slice. If *chunk_size* is below
            ``1``; if ``k >= 2`` but the learner's ``vjp_method`` is not
            ``'multi-step'`` (the executor would raise three frames down), the
            learner is vmapped (``in_axes=0`` would map time as the batch
            axis), or ``T % k != 0``. If *mask* is not shape ``(T,)``. If
            *reduction* or *loss_output* is not one of its legal values. If
            ``step_fn`` returns a non-scalar in plain mode, or anything but
            shape ``(k,)`` in window mode. If the learner has not been
            compiled, the existing guard raises before any of these.

        Warnings
        --------
        Two known limitations are *reachable through this method*. Neither is a
        driver defect -- both belong to the engines it drives -- but
        ``chunk_size`` is what makes them easy to hit by accident, so they are
        named here rather than only in the limitations document.

        **F-30, window mode on an IO-factorized engine.** ``pp_prop`` /
        ``ES_D_RTRL`` / ``OSTLFeedforward``, and any
        ``trace_factorization='io_factorized'`` config, undo the warm-up bias
        of their f-side smoothing with a factor indexed by ``running_index``,
        which counts ``update()`` calls. A ``k``-step window advances the trace
        ``k`` times but ``running_index`` once, so the correction lags the trace
        by a factor of ``k``. Measured at ``T=6, k=2, decay=0.9``: 2.1e-03
        relative against the same run indexed by true trace steps. Driving with
        ``chunk_size=None`` avoids it entirely.

        **F-35, DNI and the synthesizer's deployment contract.** A
        :func:`train_synthetic_gradient` fit is valid only for the exact
        ``(loss_fn, chunk_size)`` pair it was trained on, and **nothing checks
        that deployment matches**. Fit and drive at the same ``chunk_size`` --
        note that ``1`` and ``None`` are the same path on both sides. A mismatch
        is not degraded DNI but noise shaped like a cotangent: fitting at ``1``
        and deploying at ``2`` moves the gradient 2.0e-02 relative. ``mask`` is
        the other half of the contract, since it changes the objective the
        synthesizer is predicting the future of. ``reduction`` is **not**: it
        divides once after the loop and never enters the differentiated
        objective.

        Notes
        -----
        ``grads`` is the learner's **online-gradient estimate** of the reduced
        objective, not in general its mathematical derivative. For an exact
        algorithm inside its valid regime the two coincide; for every
        approximate rule they deliberately do not, and that difference is the
        algorithm's content. ``reduction`` and ``mask`` define the objective the
        estimate is *aimed at*.

        Examples
        --------
        .. code-block:: python

            >>> import braintools
            >>> import braintrace
            >>> learner = braintrace.compile(model, 'D_RTRL', inputs[0], batch_size=1)
            >>> opt = braintools.optim.Adam(1e-3)
            >>> opt.register_trainable_weights(learner.param_states)
            >>>
            >>> def step_loss(inp, tar):
            ...     out = learner(inp)
            ...     return braintools.metric.squared_error(out, tar).mean()
            >>>
            >>> grads, loss = learner.etrace_grad(
            ...     inputs, targets, step_fn=step_loss,
            ...     loss_output='scalar', return_value=True)
            >>> opt.update(grads)
        """
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f'reduction must be one of {_REDUCTIONS}, got {reduction!r}.')
        if loss_output not in _LOSS_OUTPUTS:
            raise ValueError(
                f'loss_output must be one of {_LOSS_OUTPUTS}, got {loss_output!r}.')

        chunk_size = _check_chunk_size(chunk_size)
        length = _sequence_length(sequences)
        self._seq_check_window(chunk_size, length, for_grad=True)
        mask = _check_mask(mask, length)

        if weights is None:
            weights = self._seq_param_states

        windowed = chunk_size is not None
        # Branch on ``chunk_size`` rather than ``windowed`` so the int is
        # narrowed for the arithmetic below; ``windowed`` is only read as a
        # flag from the closures, which no narrowing would reach anyway.
        if chunk_size is not None:
            n_steps = length // chunk_size
            xs = (_to_windows(sequences, n_steps, chunk_size),
                  mask.reshape(n_steps, chunk_size))
        else:
            xs = (sequences, mask)

        def objective(slices, weight):
            result = step_fn(*slices)
            if has_aux:
                loss, aux = result
            else:
                loss, aux = result, None
            if windowed:
                if jnp.shape(loss) != (chunk_size,):
                    raise ValueError(
                        f'with chunk_size={chunk_size}, step_fn must return a '
                        f'({chunk_size},) vector of per-step losses, got shape '
                        f'{jnp.shape(loss)}. A window-level objective is spread '
                        f'over the window explicitly, e.g. '
                        f'jnp.broadcast_to(value / {chunk_size}, ({chunk_size},)).'
                    )
            elif jnp.ndim(loss) != 0:
                raise ValueError(
                    f'with chunk_size=None, step_fn must return a scalar loss, '
                    f'got shape {jnp.shape(loss)}. Reduce over the batch and '
                    f'feature axes inside step_fn.'
                )
            return jnp.sum(weight * loss), (loss, aux)

        grad_fn = brainstate.transform.grad(objective, weights, has_aux=True)

        def body(carry, xs_t):
            slices, weight = xs_t
            grads, (loss, aux) = grad_fn(slices, weight)
            return jax.tree.map(jnp.add, carry, grads), (loss, aux)

        init = jax.tree.map(jnp.zeros_like,
                            {k: v.value for k, v in weights.items()})
        total, (losses, aux) = brainstate.transform.scan(body, init, xs)

        # `mask` is applied inside the differentiated objective, so the
        # reduction denominator is the only thing left to apply -- once, after
        # the scan. The objective is linear in the per-step losses and the
        # gradients accumulate additively, so dividing at the end is exact
        # rather than an average of per-window means.
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        if reduction == 'mean':
            total = jax.tree.map(lambda g: g / denom, total)

        if not return_value:
            return (total, aux) if has_aux else total

        losses = losses.reshape((length,))
        masked = losses * mask
        if loss_output == 'per_step':
            value = losses
        elif loss_output == 'masked':
            value = masked
        else:
            value = jnp.sum(masked)
            if reduction == 'mean':
                value = value / denom

        return (total, value, aux) if has_aux else (total, value)

    # -- the evolution driver -----------------------------------------------

    def etrace_evolve(
        self,
        *sequences: Any,
        step_fn: Optional[Callable] = None,
        chunk_size: Optional[int] = None,
        return_outputs: bool = False,
    ):
        r"""Drive the model and the eligibility trace forward, computing no gradient.

        Hidden states and eligibility traces advance exactly as they do inside
        :meth:`etrace_grad`.

        Parameters
        ----------
        *sequences
            As in :meth:`etrace_grad`.
        step_fn : callable, optional
            ``None`` (default) calls the learner directly with the slices --
            and, under window mode, wraps them in :class:`MultiStepData`, since
            with no ``step_fn`` every sequence is by definition a model input.
            Supplying a ``step_fn`` opts out of that wrapping entirely.
        chunk_size : int, optional
            As in :meth:`etrace_grad`, **except** that windows are legal on
            either ``vjp_method``: this method runs no loss VJP, which is the
            only thing single-step learners refuse.
        return_outputs : bool, optional
            ``False`` (default) returns ``None`` and stacks nothing, so a long
            warm-up costs no output memory. ``True`` stacks whatever the call
            returned, with leading axis ``T // chunk_size``.

        Returns
        -------
        None or stacked outputs
            ``None`` when ``return_outputs=False``; otherwise the stacked
            per-step (or per-window) return values of the driven call.

        Raises
        ------
        TypeError
            As in :meth:`etrace_grad`, for *chunk_size* and for wrapped
            sequences.
        ValueError
            As in :meth:`etrace_grad`, **except** that a window is *not*
            refused for being on a single-step learner -- no loss VJP is taken
            here, so the restriction does not apply. Windows are still refused
            under a vmapped learner, and ``T % chunk_size == 0`` still holds.

        Examples
        --------
        .. code-block:: python

            >>> import braintrace
            >>> learner = braintrace.compile(model, 'D_RTRL', xs[0], batch_size=1)
            >>> warmup_inputs, xs, ys = inputs[:20], inputs[20:], targets[20:]
            >>>
            >>> learner.etrace_evolve(warmup_inputs)          # free-running prefix
            >>> grads = learner.etrace_grad(xs, ys, step_fn=step_loss)
        """
        chunk_size = _check_chunk_size(chunk_size)
        length = _sequence_length(sequences)
        self._seq_check_window(chunk_size, length, for_grad=False)

        windowed = chunk_size is not None
        if chunk_size is not None:  # narrows the int; see ``etrace_grad``
            n_steps = length // chunk_size
            xs = _to_windows(sequences, n_steps, chunk_size)
        else:
            xs = sequences

        call = self._seq_call

        def body(*slices):
            if step_fn is None:
                if windowed:
                    slices = tuple(MultiStepData(s) for s in slices)
                out = call(*slices)
            else:
                out = step_fn(*slices)
            return out if return_outputs else None

        return brainstate.transform.for_loop(body, *xs)


class ETraceVmap(SequenceDriverMixin, brainstate.nn.Vmap):
    """Provide sequence drivers on a ``brainstate.nn.Vmap`` wrapper.

    Returned by ``braintrace.compile(..., vmap=True)`` so the call site is
    identical in batched and unbatched mode. Because it *is* a
    ``brainstate.nn.Vmap``, calling it, its attributes and every
    ``isinstance(x, brainstate.nn.Vmap)`` check keep working; only the added
    methods are new.

    One thing does change: ``type(x) is brainstate.nn.Vmap`` is now ``False``,
    so a caller dispatching on the *exact* runtime type takes a different
    branch, and ``repr`` reads ``ETraceVmap``. Use ``isinstance``. (Pickling is
    unaffected -- a bare ``Vmap`` was already unpicklable, for the same
    ``weakref`` reason.)

    Reaching into ``.module`` is not an equivalent: ``learner.module.etrace_grad(...)``
    would drive the **unbatched** learner and silently produce per-lane-wrong
    results.

    Window mode is refused here -- see
    :meth:`SequenceDriverMixin.etrace_grad`.
    """
    __module__ = 'braintrace'

    _seq_is_vmapped = True

    @property
    def _seq_param_states(self):
        return self.module.param_states

    @property
    def _seq_vjp_method(self) -> Optional[str]:
        return getattr(self.module, 'vjp_method', None)
