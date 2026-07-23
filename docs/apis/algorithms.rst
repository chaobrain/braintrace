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
   :template: classtemplate.rst

   compile


Base Classes
------------

The abstract bases shared by every algorithm. :class:`ETraceAlgorithm` is the
root; :class:`ETraceVjpAlgorithm` adds the VJP-based machinery that the
concrete D-RTRL / ES-D-RTRL / SNN algorithms build on. :class:`EligibilityTrace`
is the state these algorithms carry across time.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceAlgorithm
   ETraceVjpAlgorithm
   EligibilityTrace


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

   ParamDimVjpAlgorithm
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

   IODimVjpAlgorithm
   pp_prop

:class:`pp_prop` is the concrete subclass of :class:`IODimVjpAlgorithm`;
``ES_D_RTRL`` is an alias for :class:`pp_prop`.


SNN Online-Learning Algorithms
------------------------------

Paper-faithful algorithms tailored to spiking neural networks, all
``ETraceVjpAlgorithm`` subclasses. These are **approximate** (except where a
regime makes them exact); know the regime before relying on their gradients.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EProp
   OSTLRecurrent
   OSTLFeedforward
   OTPE
   OTTT
   OSTTP

Trace helpers reused across the SNN algorithms — a frozen random-feedback
projection, an output-side low-pass filter, and a leaky presynaptic
accumulator:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FixedRandomFeedback
   KappaFilter
   PresynapticTrace


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
   * - ``OTPE``
     - :math:`O(B \cdot I \cdot O)` (full) / :math:`O(B(I+O))` (approx)
     - :math:`O(B \cdot I \cdot O)`
     - Deep SNNs; F-OTPE trades rank for memory
   * - ``OTTT``
     - :math:`O(B \cdot I)`
     - :math:`O(B \cdot I \cdot O)`
     - Very large SNNs; presynaptic λ-trace only
   * - ``OSTTP``
     - :math:`O(B \cdot |\theta|)`
     - :math:`O(B \cdot I \cdot O)`
     - Target-projection via fixed random feedback
