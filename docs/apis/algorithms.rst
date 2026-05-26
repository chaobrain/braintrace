Online Learning Algorithms
==========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1

``braintrace`` provides online learning algorithms based on
eligibility trace propagation. All algorithms share the same interface:
wrap a model, compile the graph, then call the algorithm as a drop-in
replacement for the model's forward pass.


Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceAlgorithm
   EligibilityTrace


D-RTRL (Parameter Dimension)
-----------------------------

The Decoupled Real-Time Recurrent Learning algorithm with diagonal
approximation. Memory complexity: :math:`O(B \cdot |\theta|)`, where
:math:`B` is the batch size and :math:`|\theta|` is the number of parameters.

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

``D_RTRL`` is an alias for :class:`ParamDimVjpAlgorithm`.


ES-D-RTRL (Input-Output Dimension)
------------------------------------

The Event-Synchronized D-RTRL algorithm. Factorizes the eligibility trace
into input and output components with exponential smoothing. Memory
complexity: :math:`O(B(I + O))`, where :math:`I` and :math:`O` are the
input and output dimensions.

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

``ES_D_RTRL`` and ``IODimVjpAlgorithm`` are aliases for :class:`pp_prop`.


VJP Algorithm Base
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceVjpAlgorithm


SNN Online-Learning Algorithms
------------------------------

Paper-faithful algorithms tailored to spiking neural networks. All are
``ETraceVjpAlgorithm`` subclasses (or factories over the VJP algorithms).

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

SNN helpers reusable across the above algorithms:

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
