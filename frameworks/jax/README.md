# Framework Track — JAX / Flax

Research-oriented implementations using JAX and Flax.

JAX is the framework of choice for many research labs (Google Brain, DeepMind). Its key features:
- Composable transforms: `jit`, `grad`, `vmap`, `pmap`
- XLA compilation: fast on TPUs/GPUs
- Functional style: explicit random keys, no mutable state

## Contents (planned)

| File | Topic |
|---|---|
| `attention_flax.ipynb` | Transformer in Flax |
| `custom_grad.ipynb` | Custom gradients with `jax.custom_vjp` |
| `distributed_training.ipynb` | Multi-device training with `pmap` |

## JAX Fundamentals

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax  # optimizer library

# Key patterns:
# 1. Always pass random keys explicitly
key = jax.random.PRNGKey(42)

# 2. JIT-compile for speed
@jax.jit
def forward(params, x):
    ...

# 3. Vectorize over batch with vmap
batched_forward = jax.vmap(forward, in_axes=(None, 0))
```
