import numpy as np
import autograd as ag
import jax
import jax.numpy as jnp

def f(x): return jnp.array([x,2]) + jnp.array([2.1,x])
def g(x): return jnp.array([x,2]) - jnp.array([2.1,x])

# df = ag.grad(f)
# df = ag.jacobian(f)
# print(df(2.0))

# df = jax.jacrev(f)
# print(df(0.0))

# SEE https://jax.readthedocs.io/en/latest/advanced-autodiff.html#advanced-autodiff

print('\nf:')

# forward
# result = (y, dydx)
y, dydx = jax.jvp(f, [2.0], [1.0])
print(y, dydx)

# backward
# (y, fun)
y, fp = jax.vjp(f, 2.0)
print(y,fp(jnp.array([1.0,1.0])))

print('\ng:')

# forward
# result = (y, dydx)
u, dgdx = jax.jvp(g, [2.0], [1.0])
print(u, dgdx)

# backward
# (y, fun)
u, gp = jax.vjp(g, 2.0)
res = gp(jnp.array([1.0,1.0]))
print(u,gp(jnp.array([1.0,1.0])))
print(res[0])


