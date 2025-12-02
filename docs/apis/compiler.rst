Online Learning Compiler
========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1



Data Structure
--------------

The following classes are the data structures used for storing eligibility trace graph during the compilation.
Our eligibility trace compiler needs to store the hidden state groups, the operations, the parameter weights,
the perturbations, and the compiled model.

- :class:`HiddenGroup` summarizes all hidden state groups in the model.
  Each hidden state group contains multiple hidden state, and the hidden state transition function.
- :class:`HiddenParamOpRelation` summarizes the relation between hidden state groups,
  the associated parameter weights, and the operations that use them.
- :class:`HiddenPerturbation` summarizes the perturbation of hidden state groups.
  It contains the perturbation function, and the perturbation hidden target.
- :class:`ModuleInfo` contains the information of the model, including the input, output, hidden, states, jaxpr, and many others.
- :class:`ETraceGraph` contains the compiled graph of the model, including the
  hidden state groups, the operations, the parameter weights, the jaxpr of compiled models,
  and others.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    HiddenGroup
    HiddenParamOpRelation
    HiddenPerturbation
    ModuleInfo
    ETraceGraph



Graph Executor
--------------

The following classes are the base classes for the online learning compilation.

``ETraceGraphExecutor`` is used to implement or execute the compiled eligibility trace graph.
It utilizes the compiled graph defined in the above data structure to execute the model,
compute the Jacobian of hidden-group, the Jacobian of weight-to-hidden-group,
and the gradient of the loss-to-hidden-group.

``ETraceGraphExecutor`` defines the abstract methods for the online learning graph execution.
Generally, the derived classes should implement the following methods:

- :meth:`ETraceGraphExecutor.solve_h2w_h2h_jacobian` to compute the Jacobian of hidden-group and weight-to-hidden-group.
- :meth:`ETraceGraphExecutor.solve_h2w_h2h_l2h_jacobian` to compute the Jacobian of loss-to-hidden-group, hidden-group, and weight-to-hidden-group.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceGraphExecutor


``ETraceVjpGraphExecutor`` implements the graph execution for the VJP-based online learning algorithms,
including those eligibility trace algorithms for:

- :class:`IODimVjpAlgorithm`
- :class:`ParamDimVjpAlgorithm`
- :class:`HybridDimVjpAlgorithm`


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceVjpGraphExecutor




