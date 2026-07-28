pp-prop Examples
================

The ``examples/pp_prop/`` series is the input/output-factorized online-learning
track. ``pp_prop`` is the canonical public name; ``ES_D_RTRL`` remains a
backward-compatible alias.

Follow the numbered scripts in order:

* `01-basics-lif-integrator.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/01-basics-lif-integrator.py>`__
  introduces the LIF integrator. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `02-neurons-alif-dms.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/02-neurons-alif-dms.py>`__
  applies an ALIF model to delayed matching-to-sample. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `03-neurons-gif-working-memory.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/03-neurons-gif-working-memory.py>`__
  demonstrates a GIF working-memory model. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `04-neurons-coba-ei-rsnn.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/04-neurons-coba-ei-rsnn.py>`__
  uses a conductance-based E/I recurrent SNN. API:
  :doc:`SignedWLinear </apis/generated/braintrace.nn.SignedWLinear>`.
* `05-batching-vmap.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/05-batching-vmap.py>`__
  uses per-sample ``vmap`` execution. API:
  :doc:`compile </apis/generated/braintrace.compile>`.
* `06-batching-batched.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/06-batching-batched.py>`__
  uses batched primitives. API:
  :doc:`compile </apis/generated/braintrace.compile>`.
* `07-vjp-single-step.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/07-vjp-single-step.py>`__
  demonstrates a single-step VJP. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `08-vjp-multi-step.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/08-vjp-multi-step.py>`__
  demonstrates a multi-step VJP. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `09-operator-sparse.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/09-operator-sparse.py>`__
  exercises a sparse recurrent operator. API:
  :doc:`SparseLinear </apis/generated/braintrace.nn.SparseLinear>`.
* `10-operator-lora.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/10-operator-lora.py>`__
  exercises low-rank recurrent weights. API:
  :doc:`lora_matmul </apis/generated/braintrace.lora_matmul>`.
* `11-operator-conv.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/11-operator-conv.py>`__
  exercises a convolutional SNN. API:
  :doc:`Conv2d </apis/generated/braintrace.nn.Conv2d>`.
* `12-classification-neuromorphic.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/12-classification-neuromorphic.py>`__
  provides a neuromorphic classification example. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `13-knob-decay-vs-rank.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/13-knob-decay-vs-rank.py>`__
  compares trace decay with rank. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.
* `14-knob-vjp-method-contrast.py <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/14-knob-vjp-method-contrast.py>`__
  contrasts VJP methods. API:
  :doc:`pp-prop </apis/generated/braintrace.pp_prop>`.

Read the
`pp-prop examples README <https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/README.md>`__
and the
:doc:`pp-prop tutorial </tutorials/pp_prop>` alongside the scripts. The
factorization reduces memory but introduces approximation error; its
suitability depends on the model dynamics and selected decay or rank.

Related API
-----------

* :doc:`pp-prop </apis/generated/braintrace.pp_prop>`
* :doc:`ETP Operators </apis/concepts>`
* :doc:`Algorithm Reference </apis/algorithms>`
