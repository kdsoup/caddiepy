from numpy import *
# Unlinearised differentiated program (python):
def l_diff_reverse(x1,x2,x3,x4,x5,dy): v1 = (x3*x5); v2 = (x4*x5); v3 = (v2*x5); v4 = ((2*((((x1 + (x2*x5)) + (v1*x5)) + (v3*x5)) + -sin(x5)))*dy); v5 = (v4*x5); v6 = (x2*v4); v8 = (v5*x5); v9 = (x3*v5); v10 = (v1*v4); v13 = (v8*x5); v14 = (x4*v8); v15 = (v2*v5); v16 = (v3*v4); v17 = (cos(x5)*-v4); v18 = (((v6 + (v9 + v10)) + ((v14 + v15) + v16)) + v17); return v4,v5,v8,v13,v18

