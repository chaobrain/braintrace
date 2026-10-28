ETP Operators
=============

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

   matmul
   grouped_matmul
   embedding
   einsum
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

Delay Lines and Other Non-Hidden State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An online algorithm credits a weight in two ways. The eligibility trace of each
hidden group the weight feeds carries its influence across steps *within that
group*. The learning signal then reads, in the same step, how the loss depends
on the group's new hidden values. A multi-step VJP (``vjp_method='multi-step'``)
given one step per call also reads the group's old values. Nothing else crosses
a step. A value kept in any other state, such as a delay line or ring buffer
held in a ``brainstate.ShortTermState``, is a given input to the next step. For
a single-step algorithm, a hidden state of *another* group that only a later
step reads is too.

So a weight whose influence reaches the loss only through the old value of such
a state has an online gradient of exactly zero, while its BPTT gradient is not
zero. Each VJP algorithm reports this as ``DiagnosticKind.ONLINE_GRADIENT_ZERO``:

- at compile time, as an INFO record that assumes the loss reads only the model
  outputs (the loss is not known yet);
- at the first gradient computation, as a warning, if the zero holds for the
  outputs and states the loss actually reads. A loss that reads a state after
  the step (a firing-rate regulariser, say) gives that state's new value a
  same-step path, and then nothing is reported.

The record names the weight and the states its credit crosses (``through``) and
lists the fixes that work for the algorithm in use:

- a ``ShortTermState`` line: hold it in a ``brainstate.HiddenState``. Under a
  multi-step VJP, and under ``OSTLRecurrent`` or ``SnAp``, which couple the
  states of a group, the weight then gets credit. Under ``D_RTRL``, ``EProp``
  and ``pp_prop`` it does not: their trace stays in the weight's own group, and
  the line joins that group only, so they need one of the other algorithms too;
- a hidden line under a single-step algorithm: use
  ``D_RTRL(vjp_method='multi-step')``, ``OSTLRecurrent`` or ``SnAp``;
- in every case: train the weight with BPTT.

Given ``MultiStepData`` of two or more steps per call, a multi-step VJP credits
the line within each call and drops it across calls. The gradient is then not
zero, and the gradient-time check skips such calls.

When the weight also reaches the loss within the step, its online gradient is
not zero but omits the credit through the line, and it can be strongly biased.
On a two-layer net with a one-step line plus a same-step readout of layer 1, it
pointed away from BPTT's gradient (cosine -0.92). This is an INFO record,
``DiagnosticKind.ONLINE_GRADIENT_PARTIAL``, which emits no warning.

Not checked: ``learning_signal='random_feedback'`` and ``'modulatory'``, and
random-projection traces (``UORO``), which credit weights differently. Not
detected (#216): a ring that is written and then read back in the same step,
such as ``brainstate.nn.Delay`` used as ``update(y)`` and then
``retrieve_at_step(d)``. The analysis treats the buffer as one array, so the
old slot it reads looks like this step's write.


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
steps during online learning is :class:`EligibilityTrace`. Its canonical API
page is indexed with the online-learning algorithms that create and update it.


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
