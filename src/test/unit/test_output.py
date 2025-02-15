import os
import sys
import unittest
from numpy import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import input.add as add
import input.ln as ln

class TestPyOutputFiles(unittest.TestCase):

    def test_add(self):
        # ARRANGE
        def f_diff(x,dx): return (dx + dx)
        def g_diff(x,dx): return dx,dx
        def f_diff_reverse(x,dy): return (dy + dy)
        def g_diff_reverse(x,dy1,dy2): return (dy1 + dy2)

        # ACT
        f_diff_result = f_diff(1.2, 1.0)
        f_diff_expect = 2.0
        g_diff_result = g_diff(1.2, 1.0)
        g_diff_expect = (1.0, 1.0)
        f_diff_reverse_result = f_diff_reverse(1.2, 1.0)
        f_diff_reverse_expect = 2.0
        g_diff_reverse_result = g_diff_reverse(1.2, 1.0, 1.0)
        g_diff_reverse_expect = 2.0

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        self.assertAlmostEqual(g_diff_reverse_expect, g_diff_reverse_result, 6)

    def test_cos(self):
        # ARRANGE
        def f_diff(x,dx): return (-sin(x)*dx)
        def f_diff_reverse(x,dy): return (-sin(x)*dy)

        # ACT
        f_diff_result = f_diff(1.2, 1.0)
        f_diff_expect = -0.9320390860
        f_diff_reverse_result = f_diff_reverse(1.2, 1.0)
        f_diff_reverse_expect = -0.9320390860

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)

    def test_exp(self):
        # ARRANGE
        def f_diff(x,dx): return (exp(x)*dx)
        def f_diff_reverse(x,dy): return (exp(x)*dy)

        # ACT
        f_diff_result = f_diff(1.2, 1.0)
        f_diff_expect = 3.3201169230
        f_diff_reverse_result = f_diff_reverse(1.2, 1.0)
        f_diff_reverse_expect = 3.320116923

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)

    def test_ln(self):
        # ARRANGE
        def f1_diff(x,dx): return (pow(x,-1)*dx)
        def f2_diff(x,dx): v1 = log(x); return (pow((x*v1),-1)*((dx*v1) + (x*(pow(x,-1)*dx))))
        def f3_diff(x1,x2,dx1,dx2): v2 = sin(x2); return (pow((x1*v2),-1)*((dx1*v2) + (x1*(cos(x2)*dx2))))
        def f1_diff_reverse(x,dy): return (pow(x,-1)*dy)
        def f2_diff_reverse(x,dy): v1 = log(x); v3 = (pow((x*v1),-1)*dy); return ((v3*v1) + (pow(x,-1)*(x*v3)))
        def f3_diff_reverse(x1,x2,dy): v2 = sin(x2); v4 = (pow((x1*v2),-1)*dy); v5 = (v4*v2); v6 = (cos(x2)*(x1*v4)); return v5,v6
        
        # ACT
        f1_diff_result = f1_diff(2.2, 1.0)
        f1_diff_expect = 0.4545454545
        f2_diff_result = f2_diff(2.2, 1.0)
        f2_diff_expect = 1.031045183
        f3_diff_result = f3_diff(2.2, 1.4, 1.0, 1.0)
        f3_diff_expect = 0.6270221803
        f1_diff_reverse_result = f1_diff_reverse(2.2, 1.0)
        f1_diff_reverse_expect = 0.4545454545
        f2_diff_reverse_result = f2_diff_reverse(2.2, 1.0)
        f2_diff_reverse_expect = 1.031045183
        f3_diff_reverse_result = f3_diff_reverse(2.2, 1.4, 1.0)
        f3_diff_reverse_expect = (0.4545454545, 0.1724767258)
        
        # ASSERT
        self.assertAlmostEqual(f1_diff_expect, f1_diff_result, 6)
        self.assertAlmostEqual(f2_diff_expect, f2_diff_result, 6)
        self.assertAlmostEqual(f3_diff_expect, f3_diff_result, 6)
        self.assertAlmostEqual(f1_diff_reverse_expect, f1_diff_reverse_result, 6)
        self.assertAlmostEqual(f2_diff_reverse_expect, f2_diff_reverse_result, 6)
        for i in range(len(f3_diff_reverse_expect)):
            self.assertAlmostEqual(f3_diff_reverse_expect[i], f3_diff_reverse_result[i], 6)

    def test_mul(self):
        # ARRANGE
        def f_diff(x,dx): return (3.4*dx)
        def g_diff(x,dx): v1 = (2*x); v2 = (3.4*x); return (((2*dx)*v2) + (v1*(3.4*dx)))
        def f_diff_reverse(x,dy): return (3.4*dy)
        def g_diff_reverse(x,dy): v1 = (2*x); v2 = (3.4*x); v3 = (dy*v2); v4 = (v1*dy); return ((2*v3) + (3.4*v4))
        
        # ACT
        f_diff_result = f_diff(1.5, 1.0)
        f_diff_expect = 3.4
        g_diff_result = g_diff(1.5, 1.0)
        g_diff_expect = 20.40
        f_diff_reverse_result = f_diff_reverse(1.5, 1.0)
        f_diff_reverse_expect = 3.4
        g_diff_reverse_result = g_diff_reverse(1.5, 1.0)
        g_diff_reverse_expect = 20.40 

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_reverse_expect, g_diff_reverse_result, 6)

    def test_neg(self):
        # ARRANGE
        # TODO
        
        # ACT
        
        # ASSERT
        self.assertTrue(False)

    def test_pow(self):
        # ARRANGE
        def f_diff(x,dx): return ((3*pow(x,2))*dx)
        def g_diff(x,dx): return ((2.5*pow(pow(x,3),1.5))*((3*pow(x,2))*dx))
        def f_diff_reverse(x,dy): return ((3*pow(x,2))*dy)
        def g_diff_reverse(x,dy): return ((3*pow(x,2))*((2.5*pow(pow(x,3),1.5))*dy))
        
        # ACT
        f_diff_result = f_diff(1.3, 1.0)
        f_diff_expect = 5.07
        g_diff_result = g_diff(1.6, 1.0)
        g_diff_expect = 159.1626460
        f_diff_reverse_result = f_diff_reverse(1.3, 1.0)
        f_diff_reverse_expect = 5.07
        g_diff_reverse_result = g_diff_reverse(1.6, 1.0)
        g_diff_reverse_expect = 159.1626460

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_reverse_expect, g_diff_reverse_result, 6)

    def test_proj(self):
        # ARRANGE
        def f_diff(x1,x2,dx1,dx2): return (dx1 + dx2)
        def g_diff(x1,x2,x3,dx1,dx2,dx3): return (dx1 + dx2),dx3
        def f_diff_reverse(x1,x2,dy): return dy,dy
        def g_diff_reverse(x1,x2,x3,dy1,dy2): return dy1,dy1,dy2
        
        # ACT
        f_diff_result = f_diff(1.3, 2.6, 1.0, 1.0)
        f_diff_expect = 2.0
        g_diff_result = g_diff(1.3, 2.6, 3.0, 1.0, 1.0, 1.0)
        g_diff_expect = (2.0, 1.0)
        f_diff_reverse_result = f_diff_reverse(1.3, 2.6, 1.0)
        f_diff_reverse_expect = (1.0, 1.0)
        g_diff_reverse_result = g_diff_reverse(1.3, 2.6, 3.0, 1.0, 1.0)
        g_diff_reverse_expect = (1.0, 1.0, 1.0)

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_reverse_expect, g_diff_reverse_result, 6)

    def test_simple1(self):
        # ARRANGE
        def f_diff(x,dx): return (pow(sin(x),-1)*(cos(x)*dx))
        def f_diff_reverse(x,dy): return (cos(x)*(pow(sin(x),-1)*dy))
        
        # ACT
        f_diff_result = f_diff(2.3, 1.0)
        f_diff_expect = -0.8934844633
        f_diff_reverse_result = f_diff_reverse(2.3, 1.0)
        f_diff_reverse_expect = -0.8934844633

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)

    def test_simple2(self):
        # ARRANGE
        def f_diff(x1,x2,dx1,dx2): return (dx1 + dx2)
        def f_diff_reverse(x1,x2,dy): return dy,dy
        
        # ACT
        f_diff_result = f_diff(2.3, 4.1, 1.0, 1.0)
        f_diff_expect = 2.0
        f_diff_reverse_result = f_diff_reverse(2.3, 4.1, 1.0)
        f_diff_reverse_expect = (1.0, 1.0)

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)

    def test_simple3(self):
        # ARRANGE
        def f_diff(x,dx): return ((2*dx) + -(cos(x)*dx))
        def f_diff_reverse(x,dy): v1 = (dy*x); v2 = (2*dy); v3 = (cos(x)*-dy); v4 = (v2 + v3); return v4

        # ACT
        f_diff_result = f_diff(2.3, 1.0)
        f_diff_expect = 2.666276021
        f_diff_reverse_result = f_diff_reverse(2.3, 1.0)
        f_diff_reverse_expect = 2.666276021

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)

    def test_sin(self):
        # ARRANGE
        def f_diff(x,dx): return (cos(x)*dx)
        def g_diff(x1,x2,dx1,dx2): return -(cos(x1)*dx1)
        def f_diff_reverse(x,dy): return (cos(x)*dy)
        def g_diff_reverse(x1,x2,dy): v1 = (cos(x1)*-dy); return v1,0

        # ACT
        f_diff_result = f_diff(1.1, 1.0)
        f_diff_expect = 0.4535961214
        f_diff_reverse_result = f_diff_reverse(1.1, 1.0)
        f_diff_reverse_expect = 0.4535961214
        g_diff_result = g_diff(1.1, 2.1, 1.0, 1.0)
        g_diff_expect = -0.4535961214
        g_diff_reverse_result = g_diff_reverse(1.1, 2.1, 1.0)
        g_diff_reverse_expect = (-0.4535961214, 0)

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        for i in range(len(g_diff_reverse_expect)):
            self.assertAlmostEqual(g_diff_reverse_expect[i], g_diff_reverse_result[i], 6)
        
    def test_sub(self):
        # ARRANGE
        def f_diff(x,dx): return dx
        def g_diff(x,dx): return dx,-dx
        def f_diff_reverse(x,dy): return dy
        def g_diff_reverse(x,dy1,dy2): return (dy1 + -dy2)

        # ACT
        f_diff_result = f_diff(4.5, 1.0)
        f_diff_expect = 1.0
        f_diff_reverse_result = f_diff_reverse(4.5, 1.0)
        f_diff_reverse_expect = 1.0
        g_diff_result = g_diff(4.5, 1.0)
        g_diff_expect = (1.0, -1.0)
        g_diff_reverse_result = g_diff_reverse(4.5, 1.0, 1.0)
        g_diff_reverse_expect = 0

        # ASSERT
        self.assertAlmostEqual(f_diff_expect, f_diff_result, 6)
        self.assertAlmostEqual(f_diff_reverse_expect, f_diff_reverse_result, 6)
        self.assertAlmostEqual(g_diff_expect, g_diff_result, 6)
        self.assertAlmostEqual(g_diff_reverse_expect, g_diff_reverse_result, 6)

if __name__ == '__main__':
    unittest.main()