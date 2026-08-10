# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""TurboQuant vector quantization primitives for eligibility-trace workloads."""

from ._hadamard import (
    block_hadamard_matrix,
    rotate_blocks,
    unrotate_blocks,
    sign_diagonal,
)
from ._lloydmax import (
    LLOYDMAX_CENTROIDS,
    lloydmax_codebook,
    encode_nearest,
    decode_centroids,
)
from ._turboquant import (
    TurboQuantSpec,
    TurboQuantCode,
    build_spec,
    encode,
    decode,
    relative_distortion,
)

__all__ = [
    'block_hadamard_matrix',
    'rotate_blocks',
    'unrotate_blocks',
    'sign_diagonal',
    'LLOYDMAX_CENTROIDS',
    'lloydmax_codebook',
    'encode_nearest',
    'decode_centroids',
    'TurboQuantSpec',
    'TurboQuantCode',
    'build_spec',
    'encode',
    'decode',
    'relative_distortion',
]
