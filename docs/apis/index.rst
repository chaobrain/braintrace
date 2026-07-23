API Reference
=============

``braintrace`` trains recurrent and spiking neural networks **online** —
forward in time, without backpropagation through time (BPTT). You mark the
trainable operations of a model with **ETP primitives** (for example
:func:`braintrace.matmul` instead of ``x @ w``); a compiler then discovers how
each parameter influences the network's hidden states and wires up the
eligibility traces that carry gradient information forward.

This reference is organized around the four layers of the package, with
dependencies pointing strictly downward, plus the ready-made neural-network
layers built on top.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Layer
     - What it does
     - Reference
   * - **Operators**
     - User-facing ETP ops that mark weights for online learning, and the
       machinery to register your own primitives.
     - :doc:`concepts`, :doc:`primitives`
   * - **Compiler**
     - Walks the JAX ``jaxpr``, identifies ETP primitives by type, and connects
       each parameter to the hidden states it influences.
     - :doc:`compiler`
   * - **Executor**
     - Runs the forward pass and computes the hidden→weight / hidden→hidden
       Jacobians the algorithms consume.
     - :doc:`compiler`
   * - **Algorithms**
     - Online-learning estimators: D-RTRL, pp-prop (historically ES-D-RTRL),
       and the SNN algorithms (EProp, OSTL, OTPE, OTTT, OSTTP).
     - :doc:`algorithms`
   * - **Layers**
     - Drop-in ``brainstate.nn``-style layers pre-wired through ETP primitives.
     - :doc:`nn`

The fastest way in is the one-call :func:`braintrace.compile` entry point,
documented in :doc:`algorithms`.

.. toctree::
   :maxdepth: 2

   concepts
   primitives
   compiler
   algorithms
   nn
