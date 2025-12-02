Online Learning Concepts
========================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 1


ETrace State
------------


If you are trying to define the hidden states for eligibility trace-based learning,
you can use the following classes to define the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceState
    ETraceGroupState
    ETraceTreeState


ETrace Parameter
----------------


If you are trying to define the weight parameters for eligibility trace-based learning,
you can use the following classes to define the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceParam
    ElemWiseParam



If you do not want to compute weight gradients using eligibility trace-based learning,
you can use :py:class:`NonTempParam`, which computes the gradients using the standard
backpropagation algorithm at the current time step, while it is satisfying the
same interface as :py:class:`ETraceParam`.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    NonTempParam




If you do not want to compute weight gradients at all, you can use :py:class:`FakeETraceParam`,
or :py:class:`FakeElemWiseParam`, which does not compute the gradients at all.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    FakeETraceParam
    FakeElemWiseParam



ETrace Operator
---------------


Eligibility trace-based operators define the operations that transform the inputs and the weights
to the outputs.



.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceOp
    MatMulOp
    ElemWiseOp
    ConvOp
    SpMatMulOp
    LoraOp



ETrace Input Data
-----------------

The input data for eligibility trace-based learning should be in the form of
:class:`SingleStepData` or :class:`MultiStepData`.



.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    SingleStepData
    MultiStepData

