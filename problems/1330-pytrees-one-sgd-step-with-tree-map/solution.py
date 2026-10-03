import jax
import jax.numpy as jnp

def sgd_step(params, grads, lr):
    """Return a new PyTree: params - lr * grads, leaf-wise (params unchanged)."""    
    return jax.tree.map(lambda params, grads: params - lr * grads, params, grads)
