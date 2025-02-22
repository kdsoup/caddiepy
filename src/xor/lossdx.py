from numpy import *

# Unlinearised differentiated program (python):
def sigmoid_diff_reverse(x1,x2,dy): v4 = ((-1*pow((1 + exp((-1*x2))),-2))*dy); v5 = (exp((-1*x2))*v4); v6 = (-1*v5); return 0,v6

def sqloss_diff_reverse(x1,x2,x3,x4,x5,x6,dy): v1 = ((x1*x5) + x3); v2 = (v1[1]); v3 = pow((1 + exp((-1*v2))),-1); v7 = ((2*(((x2*v3) + x4) + -x6))*dy); v8 = (v7*v3); v9 = ((-1*pow((1 + exp((-1*v2))),-2))*(x2*v7)); v10 = (exp((-1*v2))*v9); v11 = (-1*v10); v12 = (0,v11*x5); v13 = (x1*0,v11); v14 = -v7; return v12,v8,0,v11,v7,v13,v14

