# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r""":class:`ETPPrimitiveSpec` and the spec-based registration entry point.

A spec is a single declarative record carrying every datum the compiler
and runtime need: the primitive name, the implementation function, the
position of weight / x / y in the equation's invars / outvars, two
behaviour flags, and the four ETP rule callables. One call to
:func:`register_primitive_spec` then creates the primitive and populates
every relevant registry.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from braintrace._compatible_imports import Primitive

from ._primitive import ETPPrimitive, register_primitive

__all__ = [
    'ETPPrimitiveSpec',
    'ETP_PRIMITIVE_SPECS',
    'register_primitive_spec',
    'get_primitive_spec',
]

ETP_PRIMITIVE_SPECS: Dict[Primitive, 'ETPPrimitiveSpec'] = {}


@dataclass(frozen=True)
class ETPPrimitiveSpec:
    """Declarative specification of an ETP primitive.

    Attributes:
        name: Primitive name (e.g. ``'etp_mm'``).
        impl: Implementation function. All standard JAX rules
            (abstract_eval, lowering, JVP, transpose, batching) are
            auto-derived from this.
        yw_to_w: D-RTRL trace propagation rule.
        xy_to_dw: Weight-gradient rule.
        init_drtrl: D-RTRL parameter-dimension trace initialiser.
        init_pp: pp_prop IO-dimension df trace initialiser.
        trainable_invars_fn: Function ``eqn.params -> {key: invar_index}``
            declaring the primitive's full trainable-input layout. Used by
            the compiler and executors to support N-trainable-input primitives
            (e.g. ``{weight, bias}`` for Linear, ``{B, A, bias}`` for LoRA).
        x_invar_index: Position of the input ``x`` in ``eqn.invars``, or
            ``None`` for primitives that have no external input (currently
            only ``etp_elemwise_p``).
        y_outvar_index: Position of the output ``y`` in ``eqn.outvars``.
            All current primitives have a single output at index 0.
        batched: Whether the primitive operates on batched inputs.
        gradient_enabled: If True, the compiler *may* traverse this primitive
            when walking ``y -> h`` (identity-like ops). If False (default
            for any trainable op), the primitive acts as a tail boundary —
            a preceding ETP weight whose only path to ``h`` passes through
            this primitive is excluded from ETP.
    """

    name: str
    impl: Callable
    yw_to_w: Callable
    xy_to_dw: Callable
    init_drtrl: Callable
    init_pp: Callable
    trainable_invars_fn: Callable[[dict], Dict[str, int]]
    x_invar_index: Optional[int] = 0
    y_outvar_index: int = 0
    batched: bool = False
    gradient_enabled: bool = False

    def resolve_trainable_invars(self, eqn_params: dict) -> Dict[str, int]:
        """Return ``{key: invar_index}`` for this equation."""
        return self.trainable_invars_fn(eqn_params)


def register_primitive_spec(spec: ETPPrimitiveSpec) -> ETPPrimitive:
    """Create an :class:`ETPPrimitive` from *spec* and install every rule.

    Records *spec* in :data:`ETP_PRIMITIVE_SPECS` so the compiler can query
    the primitive's invar layout without hard-coding identity checks.
    """
    p = register_primitive(
        spec.name,
        spec.impl,
        batched=spec.batched,
        gradient_enabled=spec.gradient_enabled,
    )
    p.register_etp_rules(
        yw_to_w=spec.yw_to_w,
        xy_to_dw=spec.xy_to_dw,
        init_drtrl=spec.init_drtrl,
        init_pp=spec.init_pp,
    )
    ETP_PRIMITIVE_SPECS[p] = spec
    return p


def get_primitive_spec(primitive: Primitive) -> Optional[ETPPrimitiveSpec]:
    """Return the :class:`ETPPrimitiveSpec` for *primitive*.

    Returns ``None`` if the primitive was registered through the
    ``register_primitive`` + manual ``register_*`` API without a spec.
    """
    return ETP_PRIMITIVE_SPECS.get(primitive)
