Online-Learning Algorithms
==========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1

``braintrace`` provides online-learning algorithms based on eligibility-trace
propagation. They all share one interface: wrap a model, compile its graph,
then call the learner as a drop-in replacement for the model's forward pass —
gradients are accumulated forward in time instead of by BPTT.

Two correctness classes appear below. **Exact** algorithms compute the same
total gradient as BPTT (just forward); they match a BPTT oracle element-wise.
**Approximate** algorithms deliberately drop or factor part of the computation
and match BPTT only in the regime their math guarantees.


One-Call Entry Point
--------------------

:func:`compile` is the recommended starting point. It constructs an algorithm
for a model and eagerly builds its eligibility-trace graph, returning a
ready-to-``update`` learner in a single call.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compile


Driving a Sequence
------------------

Every learner carries two sequence drivers, so the scan-accumulate block that
online learning used to require is not something you write by hand:

.. code-block:: python

    def step_loss(inp, tar):
        return braintools.metric.squared_error(learner(inp), tar).mean()

    # optional warm-up: advance hidden states and eligibility traces, no gradient
    learner.etrace_evolve(inputs[:n_warmup])

    grads, step_losses = learner.etrace_grad(
        inputs[n_warmup:], targets, step_fn=step_loss, return_value=True)
    opt.update(grads)

:meth:`~SequenceDriverMixin.etrace_grad` owns the loop, the accumulation, the loss
mask and the reduction; ``step_fn`` owns the model call. That split is what lets
a multi-head model, a hidden-state regularizer, or a windowed objective work
without the driver knowing anything about them. Both methods are
*continuations*: they leave the final state installed, so consecutive calls
compose into one trajectory and no call implies a reset.

A ``mask`` gates the **loss only** — the learner is still driven at every step,
so a zero-weighted prefix is exactly equivalent to ``etrace_evolve`` over it.
``chunk_size=k >= 2`` hands ``step_fn`` a ``(k, ...)`` window instead of one
step and requires ``vjp_method='multi-step'``; ``chunk_size=1`` is the plain
single-step path, matching :func:`train_synthetic_gradient`'s encoding.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SequenceDriverMixin
   ETraceVmap


Axis Coordinates
----------------

The named algorithms below are *coordinates* in a six-axis space, not separate
implementations. :class:`ETraceConfig` names that space explicitly, so a rule
with no preset name is as constructible as one with a name:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Axis
     - Values
   * - ``trace_factorization``
     - ``'per_param'`` (:math:`O(P \cdot H)`), ``'io_factorized'`` (:math:`O(I+O)`)
   * - ``temporal_recursion``
     - ``'jacobian'``, ``'scalar_leak'``, ``'none'`` — a ``(x, f)`` pair under ``'io_factorized'``
   * - ``recurrence_scope``
     - ``'diagonal'``, ``'coupled'``
   * - ``learning_signal``
     - ``'symmetric'``, ``'random_feedback'``
   * - ``trace_filter``
     - ``'none'``, ``'kappa'``
   * - ``update_schedule``
     - ``'per_step'``

Illegal combinations are rejected at construction with an error naming the
legal pairings, and coordinates that denote the same rule are canonicalised to
one form (a zero decay, for instance, collapses to ``temporal_recursion='none'``).
Pass a config wherever :func:`compile` accepts an algorithm name:

.. code-block:: python

    # an x-side leak with an instantaneous f-side
    learner = braintrace.compile(
        model,
        braintrace.ETraceConfig(trace_factorization='io_factorized',
                                temporal_recursion=('scalar_leak', 'none'),
                                decay=(0.9, 0.0)),
        x0,
    )

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceConfig


Base Classes
------------

The abstract bases and reusable estimator engines shared across algorithms.
:class:`ETraceAlgorithm` is the root, :class:`ETraceVjpAlgorithm` adds the
VJP-based machinery, and :class:`EligibilityTrace` is the state carried across
time. The three estimator bases implement the parameter-dimensional,
input/output-factorized, and random-projection trace representations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceAlgorithm
   ETraceVjpAlgorithm
   EligibilityTrace
   ParamDimVjpAlgorithm
   IODimVjpAlgorithm
   RandomProjectionVjpAlgorithm


D-RTRL — Parameter-dimensional estimator
-----------------------------------------

Diagonal Real-Time Recurrent Learning uses a diagonal approximation of the
hidden-to-hidden Jacobian. Memory complexity is
:math:`O(B \cdot |\theta|)`, where :math:`B` is the batch size and
:math:`|\theta|` the number of parameters. It is not generally
gradient-equivalent to BPTT outside the assumptions of that approximation.

.. math::

   \boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}
   + \operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t

.. math::

   \nabla_{\boldsymbol{\theta}} \mathcal{L}
   = \sum_{t' \in \mathcal{T}} \frac{\partial \mathcal{L}^{t'}}{\partial \mathbf{h}^{t'}}
   \circ \boldsymbol{\epsilon}^{t'}

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   D_RTRL

:class:`D_RTRL` is the concrete, ready-to-use subclass of
:class:`ParamDimVjpAlgorithm`.


pp-prop — Input/output-factorized estimator
-------------------------------------------

``pp_prop`` (historically exposed as ``ES_D_RTRL``) factorizes the eligibility
trace into input and output components with exponential smoothing, reducing
memory to :math:`O(B(I + O))`, where :math:`I` and :math:`O` are the input and
output dimensions. An integer ``decay_or_rank`` value parameterizes the decay;
it does not allocate multiple rank factors.

.. math::

   \boldsymbol{\epsilon}^t \approx \boldsymbol{\epsilon}_{\mathbf{f}}^t
   \otimes \boldsymbol{\epsilon}_{\mathbf{x}}^t

.. math::

   \boldsymbol{\epsilon}_{\mathbf{x}}^t
   = \alpha \boldsymbol{\epsilon}_{\mathbf{x}}^{t-1} + \mathbf{x}^t

.. math::

   \boldsymbol{\epsilon}_{\mathbf{f}}^t
   = \alpha \operatorname{diag}(\mathbf{D}^t) \circ \boldsymbol{\epsilon}_{\mathbf{f}}^{t-1}
   + (1 - \alpha) \operatorname{diag}(\mathbf{D}_f^t)

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pp_prop

:class:`pp_prop` is the concrete subclass of :class:`IODimVjpAlgorithm`;
``ES_D_RTRL`` is an alias for :class:`pp_prop`.


SnAp — Sparse n-step RTRL approximation
---------------------------------------

:class:`SnAp` retains the recurrent influence entries reachable within an
``n``-step position neighbourhood. It interpolates from the coupled,
within-position scope at ``n=1`` toward full within-group RTRL as the
neighbourhood saturates. Dense recurrence therefore saturates immediately;
the method is most useful when the recurrent position graph is structurally
sparse.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SnAp


UORO — Random-projection estimator
----------------------------------

:class:`UORO` carries a rank-one random projection of the full recurrent
Jacobian. The projection is an unbiased estimator of the RTRL trace, trading
variance for linear carrier storage. Its reusable engine,
:class:`RandomProjectionVjpAlgorithm`, is listed under Base Classes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   UORO


Three-Factor And Bootstrapped Signals
-------------------------------------

:class:`ThreeFactor` replaces the symmetric hidden-state learning signal with
a user-supplied modulatory signal. :class:`DNI` uses a learned synthetic
gradient to carry credit across finite online windows;
:class:`SyntheticGradient` is its predictor module, and
:func:`train_synthetic_gradient` updates that predictor.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ThreeFactor
   DNI
   SyntheticGradient

.. autosummary::
   :toctree: generated/
   :nosignatures:

   train_synthetic_gradient


.. _e-prop:

E-prop — Spiking eligibility-propagation estimator
--------------------------------------------------

:class:`EProp` implements eligibility propagation for recurrent spiking neural
networks, with optional kappa filtering and fixed random-feedback learning
signals.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EProp


.. _ostl:

OSTL — Recurrent and feedforward estimators
-------------------------------------------

Online Spatio-Temporal Learning exposes recurrent (with-H) and feedforward
(without-H) regimes as separate concrete classes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   OSTLRecurrent
   OSTLFeedforward


SNN Helpers
-----------

Reusable support types for SNN learning signals: a frozen random-feedback
projection and an output-side low-pass filter.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FixedRandomFeedback
   KappaFilter


Algorithm Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Algorithm
     - Memory
     - Computation
     - Best For
   * - ``D_RTRL``
     - :math:`O(B \cdot |\theta|)`
     - :math:`O(B \cdot I \cdot O)`
     - RNNs, general-purpose
   * - ``ES_D_RTRL``
     - :math:`O(B(I + O))`
     - :math:`O(B \cdot I \cdot O)`
     - Large SNNs, memory-constrained
   * - ``EProp``
     - :math:`O(B \cdot |\theta|)`
     - :math:`O(B \cdot I \cdot O)`
     - SNNs with κ-filtered / random-feedback learning signals
   * - ``OSTLRecurrent`` / ``OSTLFeedforward``
     - depends on regime
     - depends on regime
     - ``OSTLRecurrent`` ('with-H', D-RTRL) keeps the recurrent Jacobian; ``OSTLFeedforward`` ('without-H', pp_prop) drops it.
   * - ``SnAp``
     - depends on the retained ``n``-step neighbourhood
     - depends on recurrent graph sparsity and ``n``
     - Recurrent position graphs whose structural sparsity remains useful over the requested neighbourhood.

Each named algorithm above is a preset over an :class:`ETraceConfig`; the axes
that distinguish them are:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Algorithm
     - Coordinates (fields left at their default are omitted)
   * - ``D_RTRL``
     - the default coordinate
   * - ``pp_prop`` / ``ES_D_RTRL``
     - ``trace_factorization='io_factorized'``, ``decay=<decay_or_rank>``
   * - ``EProp``
     - ``trace_filter='kappa'``, ``kappa=<kappa_filter_decay>``; with
       ``feedback='random'``, ``learning_signal='random_feedback'``
   * - ``OSTLRecurrent``
     - ``recurrence_scope='coupled'``
   * - ``OSTLFeedforward``
     - ``trace_factorization='io_factorized'``, ``decay=1e-6`` (by default)
   * - ``SnAp``
     - ``recurrence_scope='sparse_n'``, ``sparse_n=n``

Because these are coordinates rather than classes, the axes compose beyond the
named presets — random feedback on the :math:`O(I+O)` trace, or a coupled
recurrence scope with an ``io_factorized`` factorization, are both reachable
even though no preset spells them.
