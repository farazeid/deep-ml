import jax
import jax.numpy as jnp

def sample_pair(seed, shape):
    """Return (a, b): two different standard-normal arrays of `shape`,
    drawn from two subkeys split off PRNGKey(seed)."""
    rng = jax.random.key(seed)
    rng1, rng2 = jax.random.split(rng, 2)
    return jax.random.normal(rng1, shape=(shape)), jax.random.normal(rng2, shape=(shape))
