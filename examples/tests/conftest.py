"""Test isolation for the example smoke / compile-mode suites.

Running many heavy ``jit`` + ``grad`` example cycles back-to-back in a single
pytest process accumulates global JAX state. In particular, the example models
build parameters via ``brainstate.random`` initializers, which call
``brainstate.transform.jit_error_if`` (bounds checks such as ``low < high`` in
``truncated_normal``). ``jit_error_if`` lowers to ``jax.debug.callback(...,
ordered=True)``, an *ordered effect* whose runtime token is threaded through the
process-global ``jax._src.dispatch.runtime_tokens`` set.

When a later jitted/donating computation reuses or releases the buffer backing a
token that is still recorded in ``runtime_tokens``, that token reference becomes
a *deleted/donated* buffer. The next ordered-effect op then calls
``block_until_ready()`` on the dangling token and raises::

    JaxRuntimeError: INVALID_ARGUMENT: BlockHostUntilReady() called on deleted
    or donated buffer

This only surfaces once enough heavy cases have run before a token-using case in
the same process (e.g. the ``test_compile_modes`` cases ahead of the ``100``/
``101`` smoke cases), which is why each file passes in isolation but the combined
run fails.

The fix is to reset the ordered-effect runtime-token set around every test so no
stale token can leak across cases. This touches a JAX internal
(``jax._src.dispatch.runtime_tokens``); the import is guarded so the fixture
degrades to a no-op if that internal is ever renamed, rather than breaking
collection.
"""

import gc

import pytest

try:  # pragma: no cover - guarded against JAX internal API drift
    from jax._src import dispatch as _jax_dispatch
except Exception:  # pragma: no cover
    _jax_dispatch = None


def _reset_runtime_tokens() -> None:
    """Drop any ordered-effect tokens left in JAX's process-global token set.

    Clearing the dicts is safe: it does not touch live buffers, it only forgets
    stale token references. The next ordered-effect op re-establishes a fresh
    token. Done before *and* after each test so a token donated/deleted by one
    case cannot poison the next.
    """
    tokens = getattr(_jax_dispatch, "runtime_tokens", None)
    clear = getattr(tokens, "clear", None)
    if callable(clear):
        clear()


@pytest.fixture(autouse=True)
def _isolate_jax_runtime_tokens():
    _reset_runtime_tokens()
    yield
    _reset_runtime_tokens()
    gc.collect()
