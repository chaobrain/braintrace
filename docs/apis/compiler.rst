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


Compilation Report
------------------

:class:`CompilationReport` is the structured summary attached to every learner
returned by :func:`compile`. Access it via ``learner.report`` after compiling a
model. It aggregates the diagnostics, counts, and graph information produced
during compilation into a single inspectable object.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CompilationReport

**Key members:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Description
   * - ``counts``
     - Dict of summary counts: number of hidden groups, ETP relations, etrace
       weights, excluded weights, and dynamic states.
   * - ``diagnostics``
     - Sequence of :class:`CompilationRecord` objects, one per compiler
       decision (inclusions, exclusions, warnings, errors).
   * - ``dynamic_states``
     - List of hidden-state paths discovered by the compiler.
   * - ``etrace_weights``
     - List of weight paths that participate in online learning (have ETP
       relations).
   * - ``excluded_weights``
     - List of weight paths excluded from online learning (e.g., weights that
       only reach a hidden state through another trainable ETP primitive).
   * - ``graph``
     - The underlying :class:`ETraceGraph` object.
   * - ``hidden_groups``
     - Sequence of :class:`HiddenGroup` objects discovered by the compiler.
   * - ``show(level)``
     - Print a human-readable summary at the given verbosity level (0–2).
   * - ``to_str(...)``
     - Return the summary as a string.

**Usage example:**

.. code-block:: python

   import braintrace

   learner = braintrace.compile(model, braintrace.D_RTRL, x0, batch_size=1, verbose=2)

   # Show a summary at level 1 (groups + weight lists, no raw diagnostics)
   learner.report.show(1)

   # Inspect counts programmatically
   print(learner.report.counts)
   # e.g. {'hidden_groups': 1, 'etrace_relations': 2, 'etrace_weights': 2,
   #        'excluded_weights': 1, 'dynamic_states': 1}

   # Iterate diagnostics to find warnings
   from braintrace import DiagnosticLevel
   warnings = [d for d in learner.report.diagnostics
               if d.level == DiagnosticLevel.WARNING]
