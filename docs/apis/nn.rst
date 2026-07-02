Neural-Network Layers
=====================

.. currentmodule:: braintrace.nn

``braintrace.nn`` mirrors a subset of ``brainstate.nn``, but routes each
layer's trainable forward pass through ETP primitives. As a result, the layers
are drop-in replacements whose parameters **automatically participate in online
learning** — no manual wiring required.

For example, :class:`Linear` uses :func:`braintrace.matmul` internally, so its
weight is included in eligibility-trace computation; :class:`GRUCell` uses
:func:`braintrace.matmul` for its recurrent maps and :func:`braintrace.element_wise`
for gate operations.

.. note::

   Activation, normalization, and pooling layers are intentionally **not**
   re-implemented here. Accessing them through ``braintrace.nn`` (e.g.
   ``braintrace.nn.LayerNorm``) emits a :class:`DeprecationWarning` and forwards
   to ``brainstate.nn`` / ``brainstate.state``; use those packages directly.


Linear Layers
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   GroupedLinear
   SignedWLinear
   SparseLinear
   LoRA


Embedding Layers
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Embedding


Convolutional Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Conv1d
   Conv2d
   Conv3d


Recurrent Layers
----------------

Single-step recurrent cells. Each updates its hidden state in place and returns
the new hidden state (or, for :class:`LRUCell`, the projected output).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell
   MinimalRNNCell
   MiniGRU
   MiniLSTM
   LRUCell


Readout Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LeakyRateReadout
