.. role:: hidden
    :class: hidden-section
.. currentmodule:: braintrace


DNI
===

.. autoclass:: DNI(model, synthesizer=None, name=None, vjp_method='multi-step', fast_solve=True, trace_dtype=None, chunked_trace=True, control_flow=None, snap_max_jacobian_elements=16777216)
    :members: graph, report, compile_graph, show_graph, update, init_etrace_state, reset_state, get_etrace_of, attach_synthesizer, group_signal_shapes
