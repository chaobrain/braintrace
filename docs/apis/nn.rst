Neural Network Modules
======================

.. currentmodule:: braintrace.nn
.. automodule:: braintrace.nn

``braintrace.nn`` provides neural network layers that use ETP primitives
internally. These layers are drop-in replacements for standard layers
but automatically participate in online learning.

For example, ``braintrace.nn.Linear`` uses ``braintrace.matmul`` internally,
so its weight is automatically included in eligibility trace computation.
Similarly, ``braintrace.nn.GRUCell`` uses ``braintrace.matmul`` for its
recurrent weight and ``braintrace.element_wise`` for gate operations.


Linear Layers
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   SignedWLinear
   SparseLinear
   LoRA


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


Normalization Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm0d
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d
   LayerNorm
   RMSNorm
   GroupNorm


Readout Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LeakyRateReadout
