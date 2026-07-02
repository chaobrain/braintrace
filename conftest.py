# jax >= 0.9 removed the ``jax_pmap_shmap_merge`` config option, but optax
# 0.2.7 (imported transitively via braintools) still sets it at import time,
# which breaks ``import braintrace`` in such environments.  Swallow only that
# one failing update so the suite can import; all other options behave
# normally.
import jax

_orig_update = jax.config.update


def _compat_update(name, value):
    try:
        return _orig_update(name, value)
    except (AttributeError, KeyError, ValueError):
        if name == 'jax_pmap_shmap_merge':
            return None
        raise


jax.config.update = _compat_update
