import jax
import jax.numpy as jnp

def fit_line(x, y, lr, steps):
    """Gradient-descent fit of y ≈ w*x + b from w=b=0.
    Returns (w, b, final_loss) as Python floats."""
    
    def loss_fn(w, b, x, y):
        return jnp.mean(jnp.pow(w * x + b - y, 2))
    
    def step_fn(carry, _):
        w, b, x, y = carry
        loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b, x, y)
        w -= lr * dw
        b -= lr * db
        carry = w, b, x, y
        return carry, loss

    w, b = 0.0, 0.0
    (w, b, x, y), losses = jax.lax.scan(step_fn, (w, b, x, y), length=steps)

    return float(w), float(b), float(losses[-1])
