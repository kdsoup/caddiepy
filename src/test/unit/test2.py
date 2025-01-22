import os
import sys
import unittest
import jax
from jax.numpy import *
import jax.numpy as jnp
from numpy import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import output.ln_out as ln_dif

def f1(x): return jnp.log(x)
def f2(x): return jnp.log(x * jnp.log(x))
def f3(x): return jnp.log(x[0] * jnp.cos(x[1]))

class TestLn(unittest.TestCase):

    def test_f1_fwd_diff(self):
        ## ARRANGE
        point = 2.0
        tangent = 1.0

        ## ACT
        # jax forward return: (y, dydx)
        y_jax, dydx_jax = jax.jvp(f1, (point,), (tangent,))

        # cad fwd
        dydx_cad = ln_dif.f1_diff(point, tangent)

        # print('jax fwd: ', y_jax, dydx_jax)
        # print('cad fwd: ', dydx_cad)

        ## ASSERT
        # caddiepy result is same as jax result
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

    def test_f1_rev_diff(self):
        ## ARRANGE
        point = 2.0
        tangent = 1.0
        
        ## ACT
        # jax reverse result: (y, fun)
        y, fp = jax.vjp(f1, point)
        dydx_jax = fp(tangent)[0]

        # cad reverse
        dydx_cad = ln_dif.f1_diff_reverse(2.0, 1.0)

        # print('jax rev: ', dydx_jax)
        # print('cad rev: ', dydx_cad)

        ## ASSERT
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

    def test_f2_fwd_diff(self):
        ## ARRANGE
        point = 2.0
        tangent = 1.0

        ## ACT
        # jax forward return: (y, dydx)
        y_jax, dydx_jax = jax.jvp(f2, (point,), (tangent,))

        # cad fwd
        dydx_cad = ln_dif.f2_diff(point, tangent)

        # print('jax fwd: ', y_jax, dydx_jax)
        # print('cad fwd: ', dydx_cad)

        ## ASSERT
        # caddiepy result is same as jax result
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

    def test_f2_rev_diff(self):
        ## ARRANGE
        point = 3.0
        tangent = 1.0
        
        ## ACT
        # jax reverse result: (y, fun)
        y, fp = jax.vjp(f2, point)
        dydx_jax = fp(tangent)[0]

        # cad reverse
        dydx_cad = ln_dif.f2_diff_reverse(point, tangent)

        # print('jax rev: ', dydx_jax)
        # print('cad rev: ', dydx_cad)

        ## ASSERT
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

    def test_f3_fwd_diff(self):
        ## ARRANGE
        point1 = 2.0
        point2 = 3.0
        tangent = 1.0

        ## ACT
        # jax forward return: (y, dydx)
        y_jax, dydx_jax = jax.jvp(f3, ((point1, point2),), ((tangent,tangent),))

        # cad fwd
        dydx_cad = ln_dif.f3_diff(point1, point2, tangent, tangent)

        # print('jax fwd: ', y_jax, dydx_jax)
        # print('cad fwd: ', dydx_cad)

        ## ASSERT
        # caddiepy result is same as jax result
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

    def test_f3_rev_diff(self):
        ## ARRANGE
        point1 = 2.0
        point2 = 3.0
        tangent = 1.0
        
        ## ACT
        # jax reverse result: (y, fun)
        y, fp = jax.vjp(f3, (point1, point2))
        dydx_jax = fp(tangent)[0]

        # cad reverse
        dydx_cad = ln_dif.f3_diff_reverse(point1, point2, tangent)

        # print('jax rev: ', dydx_jax)
        # print('cad rev: ', dydx_cad)

        ## ASSERT
        self.assertEqual(
            dydx_cad,
            dydx_jax
        )

if __name__ == '__main__':
    unittest.main()