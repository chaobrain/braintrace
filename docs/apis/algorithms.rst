Online Learning Algorithms
==========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1




Base Class
----------

:py:class:`ETraceAlgorithm` is the base class for all online learning algorithms. It provides the basic interface
for the learning algorithms to interact with the network. The learning algorithms are responsible for updating the
weights of the network based on the error signal. The error signal is computed by the network and passed to the
learning algorithm. The learning algorithm computes the weight updates based on the error signal and the input signal.
The weight updates are then applied to the network.


:class:`EligibilityTrace` provides the interface to store eligibility traces data for the learning algorithms.
The eligibility traces are used to compute the weight updates.



.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceAlgorithm
    EligibilityTrace



VJP Algorithms
--------------


Vector-Jacobian Product (VJP) algorithms are used to compute the weight updates for the learning algorithms
compatible with the standard VJP backpropagation algorithm, such as ``jax.grad``, ``jax.vjp``, or
``jax.jacrev`` functions.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst


    ETraceVjpAlgorithm
    IODimVjpAlgorithm
    ParamDimVjpAlgorithm
    HybridDimVjpAlgorithm
    ES_D_RTRL
    D_RTRL
    pp_prop

