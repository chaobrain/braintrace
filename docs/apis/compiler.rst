Compiler
========

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1

The compiler analyzes the model's JAX intermediate representation (Jaxpr)
to discover relationships between ETP primitives, weight parameters,
and hidden states. This page documents the compiler pipeline and its
data structures.


Graph Compilation
-----------------

The main entry point for compiling a model into an eligibility trace graph.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   compile_etrace_graph
   ETraceGraph


Module Info
-----------

Extracts the Jaxpr and state information from a ``brainstate.nn.Module``.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   extract_module_info
   ModuleInfo


Hidden Groups
-------------

Groups of hidden states that are updated together in the recurrent computation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenGroup
   find_hidden_groups_from_minfo
   find_hidden_groups_from_module


Hidden-Parameter-Operation Relations
-------------------------------------

Connections between ETP primitives, weight parameters, and hidden states.
Each relation describes: *"weight W is used through ETP primitive P,
and the output feeds into hidden group H."*

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenParamOpRelation
   find_hidden_param_op_relations_from_minfo
   find_hidden_param_op_relations_from_module


Hidden Perturbation
-------------------

Perturbation structures for computing hidden-to-hidden Jacobians
(the diagonal approximation of :math:`\partial h^t / \partial h^{t-1}`).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenPerturbation
   add_hidden_perturbation_from_minfo
   add_hidden_perturbation_in_module


Graph Executor
--------------

Executes the compiled graph: runs the forward pass and computes
the hidden-to-weight and hidden-to-hidden Jacobians.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceGraphExecutor
   ETraceVjpGraphExecutor
