import os
import sys
import unittest
from jax import jacfwd, jacrev
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import output.add as add

def f(x): return jnp.array([x,2]) + jnp.array([2.1,x])
def g(x): return jnp.array([x,2]) - jnp.array([2.1,x])

class TestAdd(unittest.TestCase):

    def test_f_fwd_diff(self):
        fdx = jacfwd(f)
        result = tuple(r.item() for r in fdx(2.0))

        self.assertEqual(
            add.f_diff(2.0,1.0),
            result
        )

    def test_g_fwd_diff(self):
        fdx = jacfwd(g)
        result = tuple(r.item() for r in fdx(2.0))

        self.assertEqual(
            add.g_diff(2.0,1.0),
            result
        )

if __name__ == '__main__':
    unittest.main()