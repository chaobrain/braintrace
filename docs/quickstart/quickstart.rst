Quickstart
==========

This example trains a small :class:`braintrace.nn.MiniGRU` on one fixed
sequence. It shows the complete online-learning path: compile the model,
differentiate each time step, scan over the sequence, update parameters, and
compare loss from clean initial states.

.. code-block:: python

   import brainstate
   import braintools
   import braintrace
   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import warnings

   # Fixed model initialization and fixed data make the result reproducible.
   brainstate.random.seed(7)

   class SequenceModel(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.rnn = braintrace.nn.MiniGRU(in_size=1, out_size=6)
           self.readout = braintrace.nn.Linear(6, 1)

       def update(self, x):
           return self.readout(self.rnn(x))

   model = SequenceModel()
   inputs = jnp.linspace(-1.0, 1.0, 12).reshape(12, 1, 1)
   targets = 0.7 * inputs + 0.2

   # Compile once. inputs[0] is one batched time step with shape (1, 1).
   with warnings.catch_warnings():
       # The readout does not feed recurrent state, so it is non-temporal.
       warnings.filterwarnings(
           "ignore",
           message=r"ETP primitive etp_mm.*has no connected hidden states.*",
       )
       learner = braintrace.compile(
           model,
           braintrace.D_RTRL,
           inputs[0],
           batch_size=1,
       )
   weights = model.states(brainstate.ParamState)
   optimizer = braintools.optim.SGD(lr=0.08)
   optimizer.register_trainable_weights(weights)

   def reset_sequence():
       # Hidden states and eligibility traces are independent state systems.
       brainstate.nn.reset_all_states(model, batch_size=1)
       learner.reset_state(batch_size=1)

   def evaluate():
       reset_sequence()
       predictions = brainstate.transform.for_loop(learner, inputs)
       return jnp.mean((predictions - targets) ** 2)

   def local_loss(x, target):
       prediction = learner(x)
       return jnp.mean((prediction - target) ** 2)

   def train_epoch(_):
       reset_sequence()

       def scan_step(accumulated_grads, sample):
           x, target = sample
           grad_fn = brainstate.transform.grad(
               local_loss, weights, return_value=True
           )
           step_grads, step_loss = grad_fn(x, target)
           accumulated_grads = jax.tree.map(
               lambda total, current: total + current,
               accumulated_grads,
               step_grads,
           )
           return accumulated_grads, step_loss

       zero_grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
       grads, step_losses = brainstate.transform.scan(
           scan_step, zero_grads, (inputs, targets)
       )
       grads = jax.tree.map(lambda grad: grad / inputs.shape[0], grads)
       optimizer.update(grads)
       return step_losses.mean()

   initial_loss = evaluate()
   training_losses = brainstate.transform.for_loop(
       train_epoch, jnp.arange(25)
   )
   final_loss = evaluate()

   print(f"initial loss: {float(initial_loss):.4f}")
   print(f"final loss: {float(final_loss):.4f}")

   with plt.style.context("default"), plt.rc_context({
       "figure.facecolor": "white",
       "axes.facecolor": "white",
       "savefig.facecolor": "white",
   }):
       fig, ax = plt.subplots(figsize=(8, 4))
       ax.plot(training_losses)
       ax.set(xlabel="Training epoch", ylabel="Mean sequence loss")
       ax.set_title("Online mini-GRU training loss")
       ax.grid(True, alpha=0.3)
       fig.tight_layout()
       fig.savefig("quickstart_loss.png", dpi=150)

With the fixed seed, the final loss should be lower than the initial loss. The
example deliberately resets both recurrent and eligibility state before each
sequence; otherwise the two loss values would not describe the same initial
condition.

.. image:: /_static/quickstart_loss.png
   :alt: Mean sequence loss across 25 online mini-GRU training epochs
   :align: center
   :width: 100%

.. centered:: *The fixed-seed loss produced by the example above.*

What happened
-------------

``braintrace.compile`` discovers which MiniGRU parameters reach recurrent
hidden state through ETP primitives. ``brainstate.transform.grad`` obtains a
per-step online gradient, while ``brainstate.transform.scan`` carries the
gradient accumulator across time. ``brainstate.transform.for_loop`` performs
the repeated updates without repeatedly dispatching model calls from Python.

Next steps
----------

* :doc:`concepts` explains how ETP primitives select trainable pathways.
* :doc:`rnn_online_learning` develops the rate-RNN workflow.
* :doc:`snn_online_learning` develops the spiking-network workflow.
