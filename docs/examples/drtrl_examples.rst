D-RTRL Examples
===============

The ``examples/drtrl/`` series demonstrates the parameter-dimensional
diagonal RTRL implementation across several model and operator families.

* `01-basics-integrator.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/01-basics-integrator.py>`__
  introduces the integrator workflow. API:
  :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`.
* `02-batching-vmap.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/02-batching-vmap.py>`__
  uses per-sample ``vmap`` execution. API:
  :doc:`compile </apis/generated/braintrace.compile>`.
* `03-batching-batched.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/03-batching-batched.py>`__
  uses batched primitives. API:
  :doc:`compile </apis/generated/braintrace.compile>`.
* `04-vjp-single-step.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/04-vjp-single-step.py>`__
  demonstrates a single-step VJP. API:
  :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`.
* `05-vjp-multi-step.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/05-vjp-multi-step.py>`__
  demonstrates a multi-step VJP. API:
  :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`.
* `07-operator-lora.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/07-operator-lora.py>`__
  covers a low-rank recurrent operator. API:
  :doc:`LoRA </apis/generated/braintrace.nn.LoRA>`.
* `08-operator-conv.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/08-operator-conv.py>`__
  covers a convolutional ETP operator. API:
  :doc:`Conv1d </apis/generated/braintrace.nn.Conv1d>`.
* `09-classification-mnist.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/09-classification-mnist.py>`__
  provides an MNIST classification task. API:
  :doc:`LSTMCell </apis/generated/braintrace.nn.LSTMCell>`.
* `10-char-lm-generation.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/10-char-lm-generation.py>`__
  provides a character-language-model task. API:
  :doc:`MiniGRU </apis/generated/braintrace.nn.MiniGRU>`.
* `11-knob-fast-solve.py <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/11-knob-fast-solve.py>`__
  examines the ``fast_solve`` implementation option. API:
  :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`.

Read the
`D-RTRL examples README <https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/README.md>`__
and the
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
