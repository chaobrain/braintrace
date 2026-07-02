ETP Operators & Core Types
==========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1

This page documents the **user-facing ETP operators** — the ops you call inside
a model's ``update`` to make a weight participate in online learning — together
with the small set of core types every algorithm consumes: input wrappers, the
eligibility-trace state, gradient utilities, and the error hierarchy.

To add your *own* ETP primitive (with custom trace-propagation rules), see
:doc:`primitives`.


ETP Primitive Operators
-----------------------

These functions mark weight operations for inclusion in online learning. Use
``braintrace.matmul(x, w)`` instead of ``x @ w`` to include a weight in
eligibility-trace computation; a parameter used through a regular JAX op is
automatically excluded. There is **no special parameter class** — every
``brainstate.ParamState`` is eligible, and participation is decided purely by
whether an ETP operator consumed it.

All operators accept physical-unit quantities (mantissa/unit are split,
computed, and recombined) and come in batched and unbatched forms selected by
input rank.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   matmul
   grouped_matmul
   embedding
   element_wise
   conv
   sparse_matmul
   lora_matmul


Controlling Parameter Participation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
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
   * - Include a parameter in online learning
     - Use a ``braintrace.*`` ETP operator (e.g. ``braintrace.matmul(x, w)``).
   * - Exclude a parameter from online learning
     - Use a regular JAX op (e.g. ``x @ w``).
   * - Selection mechanism
     - The *operation's* primitive type — not the parameter's class. Every
       ``brainstate.ParamState`` is eligible; participation depends solely on
       whether an ETP primitive consumed it.


Input Data
----------

Wrappers that tell an online-learning algorithm whether a step receives a
single time step or a whole sequence of time steps.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SingleStepData
   MultiStepData


Eligibility Trace State
-----------------------

The state object that stores the eligibility trace carried forward across time
steps during online learning.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   EligibilityTrace


Gradient Utilities
------------------

Helpers used when combining per-step gradients into the running online
gradient.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GradExpon


Errors
------

Exceptions raised by the compilation and execution machinery.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NotSupportedError
   CompilationError
