import os
import sys
import unittest
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import output.add_out as add

def f(x): return jnp.array([x,2]) + jnp.array([2.1,x])
def g(x): return jnp.array([x,2]) - jnp.array([2.1,x])

class TestAdd(unittest.TestCase):

    # def test_f_fwd_diff_1(self):
    #     fdx = jax.jacfwd(f)
    #     result = tuple(r.item() for r in fdx(2.0))

    #     self.assertEqual(
    #         add.f_diff(2.0,1.0),
    #         result
    #     )

    def test_f_fwd_diff(self):
        # jax forward return: 
        # (y, dydx)
        y, dydx = jax.jvp(f, [2.0], [1.0])

        self.assertEqual(
            add.f_diff(2.0, 1.0),
            tuple(dydx)
        )

    def test_f_rev_diff(self):
        # jax backward return: 
        # (y, fun)
        y, fp = jax.vjp(f, 2.0)

        self.assertEqual(
            add.f_diff_reverse(2.0, 1.0, 1.0),
            fp(jnp.array([1.0, 1.0]))[0]
        )

    def test_g_fwd_diff(self):
        # jax forward return: 
        # (y, dydx)
        y, dgdx = jax.jvp(g, [2.0], [1.0])

        self.assertEqual(
            add.g_diff(2.0, 1.0),
            tuple(dgdx)
        )

    def test_g_rev_diff(self):
        # jax backward return: 
        # (y, fun)
        y, gp = jax.vjp(g, 2.0)

        self.assertEqual(
            add.g_diff_reverse(2.0, 1.0, 1.0),
            gp(jnp.array([1.0, 1.0]))[0]
        )

if __name__ == '__main__':
    unittest.main()