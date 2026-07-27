Rate-Based RNN Examples
=======================

These examples focus on rate-based recurrent models and comparisons with BPTT.
The runnable scripts live in the repository's
`examples directory <https://github.com/chaobrain/braintrace/tree/main/examples>`__.

* ``100-gru-on-copying-task.py`` trains a GRU on the copying task and compares
  online diagonal RTRL with BPTT.
* ``101-integrator-rnn.py`` trains a ``MiniGRU`` integrator and plots both
  predictions and training losses.

The comparisons demonstrate behavior on the stated model and task. They do
not establish general gradient equivalence, because ``D_RTRL`` retains a
diagonal hidden-Jacobian approximation.

Begin with the :doc:`RNN Online Learning </quickstart/rnn_online_learning>`
workflow for the complete training sequence.

Related API
-----------

* :doc:`D-RTRL </apis/generated/braintrace.D_RTRL>`
* :doc:`SnAp </apis/generated/braintrace.SnAp>`
* :doc:`Neural Network Layers </apis/nn>`

