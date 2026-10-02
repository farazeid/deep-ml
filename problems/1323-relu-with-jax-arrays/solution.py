import jax
import jax.numpy as jnp

def relu(x):
    """Return x with negative entries replaced by 0 (same shape as x)."""
    return jnp.maximum(x, 0)
    def fn(a):
        return jax.lax.cond(
            a > 0,
            lambda a: a,
            lambda a: jnp.zeros_like(a),
            a,
        )
    return jax.vmap(fn)(x)