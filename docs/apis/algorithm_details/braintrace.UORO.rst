.. role:: hidden
    :class: hidden-section
.. currentmodule:: braintrace


UORO
====

.. autoclass:: UORO(model, name=None, vjp_method='multi-step', fast_solve=True, control_flow=None, projection_key=42, projection_eps=1e-12, random_feedback_key=None, snap_max_jacobian_elements=16777216)
    :members: graph, report, compile_graph, show_graph, update, init_etrace_state, reset_state, get_etrace_of
