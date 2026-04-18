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

.. list-table:: Parameter Selection Rules
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - How
   * - Include parameter in online learning
     - Use a ``braintrace.*`` ETP primitive (e.g. ``braintrace.matmul(x, w)``)
   * - Exclude parameter from online learning
     - Use a regular JAX op (e.g. ``x @ w``)
   * - Selection mechanism
     - Operation primitive type — *not* parameter class type. Every
       ``brainstate.ParamState`` is eligible; participation depends solely
       on whether an ETP primitive consumed it.


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
