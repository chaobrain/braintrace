Core Concepts
=============

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1


ETP Primitives (User API)
-------------------------

These functions mark weight operations for inclusion in online learning.
Use ``braintrace.matmul(x, w)`` instead of ``x @ w`` to include a weight
in eligibility trace computation. Parameters used with regular JAX ops
are automatically excluded — no special parameter classes needed.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   matmul
   element_wise
   conv
   sparse_matmul
   lora_matmul


Controlling Parameter Participation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import braintrace
   import brainstate

   class MyRNN(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.w_rec = brainstate.ParamState(...)   # want ETP
           self.w_in = brainstate.ParamState(...)     # do NOT want ETP
           self.h = brainstate.ShortTermState(...)

       def update(self, x):
           # regular matmul -> w_in excluded from ETP
           inp = x @ self.w_in.value
           # ETP matmul -> w_rec included in ETP
           self.h.value = jax.nn.tanh(inp + braintrace.matmul(self.h.value, self.w_rec.value))
           return self.h.value

.. list-table:: Old vs New Parameter Selection
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Old System
     - New System (ETP)
   * - Include in online learning
     - Use ``ETraceParam``
     - Use ``braintrace.matmul(x, w)``
   * - Exclude from online learning
     - Use ``brainstate.ParamState``
     - Use ``x @ w`` (regular JAX op)
   * - Selection mechanism
     - Parameter class type
     - Operation primitive type


Input Data
----------

Wrappers that tell the online learning algorithm whether the input
is a single time step or a sequence of time steps.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SingleStepData
   MultiStepData


Eligibility Trace State
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EligibilityTrace


Gradient Utilities
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GradExpon


Errors
------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NotSupportedError
   CompilationError
