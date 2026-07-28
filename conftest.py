# jax >= 0.9 removed the ``jax_pmap_shmap_merge`` config option, but optax
# 0.2.7 (imported transitively via braintools) still sets it at import time,
# which breaks ``import braintrace`` in such environments.  Swallow only that
# one failing update so the suite can import; all other options behave
# normally.
import ctypes
import gc
import os

import jax
import pytest

_orig_update = jax.config.update


def _compat_update(name, value):
    try:
        return _orig_update(name, value)
    except (AttributeError, KeyError, ValueError):
        if name == 'jax_pmap_shmap_merge':
            return None
        raise


jax.config.update = _compat_update


# JAX memoizes every executable it compiles for the life of the process and
# never evicts it, so a long-lived worker accumulates the compilation cache of
# every test it has run -- measured here at roughly 10 MB per test, climbing
# monotonically with no plateau. Spread over ~450 tests per worker that is
# several GB each, which on a 6-worker run is the difference between fitting in
# RAM and paging.
#
# Dropping the cache on a fixed cadence bounds it. The cadence trades against
# recompilation, but not monotonically -- a smaller cache also means less page
# pressure, so tightening the block buys time as well as memory until the
# recompiles start to dominate. Measured on the full suite at ``-n 6``:
#
#   ====== ======= ==========
#   block   wall    peak RSS
#   ====== ======= ==========
#   none    833 s   22.81 GB
#   100     689 s   18.50 GB
#   40      655 s   13.65 GB
#   15      684 s   11.87 GB
#   ====== ======= ==========
#
# 40 is the wall-time optimum and already cuts peak RSS by 40%. Drop it to ~15
# on a smaller machine: that is another 1.8 GB for about 4% wall time. Clearing
# after *every* test is the far end of the curve and costs 2.6x. An RSS
# threshold does not work as a trigger at all -- see the trim note below.
#
# Set BRAINTRACE_TEST_JAX_CACHE_CLEAR_EVERY=0 to disable.
_JAX_CACHE_CLEAR_EVERY = int(
    os.environ.get('BRAINTRACE_TEST_JAX_CACHE_CLEAR_EVERY', '40')
)
_tests_finished = [0]


def _make_malloc_trim():
    """Hand freed allocator arenas back to the OS, or a no-op off glibc.

    ``jax.clear_caches()`` frees the executables into the process allocator,
    but glibc keeps the emptied arenas mapped, so the clear alone moves RSS by
    nothing at all: measured 0.00 GB reclaimed by clear+collect and 0.71 GB by
    the ``malloc_trim`` immediately after it (1.49 GB -> 0.77 GB). Without this
    the cache bound is invisible to RSS and only shows up as reduced swap.

    ``malloc_trim`` is a glibc extension; on musl or macOS the lookup fails and
    the suite simply keeps the pre-trim behaviour.
    """
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
    except (OSError, AttributeError):
        return lambda: None
    return lambda: libc.malloc_trim(0)


_malloc_trim = _make_malloc_trim()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    if _JAX_CACHE_CLEAR_EVERY <= 0:
        return
    _tests_finished[0] += 1
    if _tests_finished[0] % _JAX_CACHE_CLEAR_EVERY == 0:
        jax.clear_caches()
        gc.collect()
        _malloc_trim()
