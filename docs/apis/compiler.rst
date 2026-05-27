Compiler, Executor & Diagnostics
================================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1

The compiler analyzes a model's JAX intermediate representation (``jaxpr``) to
discover the relationships between ETP primitives, weight parameters, and
hidden states. It recognizes ETP primitives by **primitive-type identity**
(never by string-matching names), and the result is an :class:`ETraceGraph`
that the executor and the online-learning algorithms consume.

Most users never call this layer directly — :func:`compile` and the algorithm
classes drive it for you. It is documented here for building custom algorithms,
inspecting what the compiler discovered, and acting on diagnostics.


Graph Compilation
-----------------

The entry point that compiles a model into an eligibility-trace graph, and the
graph object it returns.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   compile_etrace_graph
   ETraceGraph


Module Info
-----------

Extracts the ``jaxpr`` and state information from a ``brainstate.nn.Module``.
``ModuleInfo`` is the compiler's structured view of a model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   extract_module_info
   ModuleInfo


Hidden Groups
-------------

Sets of hidden states that are updated together in the recurrent computation.
The finder functions discover them either from an extracted ``ModuleInfo`` or
directly from a module.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenGroup
   find_hidden_groups_from_minfo
   find_hidden_groups_from_module


Hidden–Parameter–Operation Relations
-------------------------------------

The core data structure connecting ETP primitives, weight parameters, and
hidden states. Each relation encodes *"weight W is used through ETP primitive
P, and P's output feeds hidden group H."* Per the non-parametric-tail
invariant, a weight that reaches a hidden state only through another trainable
ETP primitive is deliberately excluded.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenParamOpRelation
   find_hidden_param_op_relations_from_minfo
   find_hidden_param_op_relations_from_module


Hidden Perturbation
-------------------

Perturbation structures used to compute hidden-to-hidden Jacobians
(the diagonal approximation of :math:`\partial \mathbf{h}^t / \partial \mathbf{h}^{t-1}`).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HiddenPerturbation
   add_hidden_perturbation_from_minfo
   add_hidden_perturbation_in_module


Graph Executor
--------------

Executes the compiled graph: runs the forward pass and computes the
hidden-to-weight and hidden-to-hidden Jacobians the algorithms consume.
:class:`ETraceVjpGraphExecutor` is the VJP-based executor used by the
``ETraceVjpAlgorithm`` family.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETraceGraphExecutor
   ETraceVjpGraphExecutor


Diagnostics
-----------

Structured, leveled records emitted while the compiler analyzes a model. They
surface issues that would otherwise be silent — for example a trainable input
that does not trace back to a ``ParamState``, or an ETP weight excluded because
it only reaches a hidden state through another trainable primitive.
:class:`DiagnosticLevel` orders records by severity (``INFO`` < ``WARNING`` <
``ERROR``) and :class:`DiagnosticKind` names the specific condition.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CompilationRecord
   DiagnosticKind
   DiagnosticLevel
