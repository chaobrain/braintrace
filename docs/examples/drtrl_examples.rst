D-RTRL Examples
===============

The ``examples/drtrl/`` series demonstrates the parameter-dimensional
diagonal RTRL implementation across several model and operator families.

* ``01`` introduces the integrator workflow.
* ``02``--``03`` cover ``vmap`` and batched execution.
* ``04``--``05`` compare single-step and multi-step VJPs.
* ``07``--``08`` cover LoRA and convolutional ETP primitives.
* ``09``--``10`` provide MNIST and character-language-model tasks.
* ``11`` examines the ``fast_solve`` implementation option.

Read ``examples/drtrl/README.md`` and the
:doc:`D-RTRL tutorial </tutorials/drtrl>` alongside the scripts. D-RTRL uses
the same ETP/compiler workflow across model architectures whose trainable
paths are expressible with registered ETP operators. This interface-level
generality does not remove the diagonal hidden-Jacobian approximation or
establish general gradient equivalence with BPTT.

Related API
-----------

* :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`
* :doc:`ETP Operators </apis/concepts>`
* :doc:`Algorithm Reference </apis/algorithms>`

