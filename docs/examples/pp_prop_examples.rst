pp-prop Examples
================

The ``examples/pp_prop/`` series is the input/output-factorized online-learning
track. ``pp_prop`` is the canonical public name; ``ES_D_RTRL`` remains a
backward-compatible alias.

Follow the numbered scripts in order:

* ``01``--``04`` introduce LIF, ALIF, GIF, and conductance-based neuron models.
* ``05``--``06`` compare per-sample ``vmap`` with batched primitives.
* ``07``--``08`` compare single-step and multi-step VJP execution.
* ``09``--``11`` exercise sparse, LoRA, and convolutional ETP primitives.
* ``12`` provides a neuromorphic classification example.
* ``13``--``14`` expose the trace-decay/rank and VJP-method trade-offs.

Read ``examples/pp_prop/README.md`` and the
:doc:`pp-prop tutorial </tutorials/pp_prop>` alongside the scripts. The
factorization reduces memory but introduces approximation error; its
suitability depends on the model dynamics and selected decay or rank.

Related API
-----------

* :doc:`pp-prop </apis/generated/braintrace.pp_prop>`
* :doc:`ETP Operators </apis/concepts>`
* :doc:`Algorithm Reference </apis/algorithms>`

