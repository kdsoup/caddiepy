import numpy as np
import lossdx

# XOR data
X = np.array([[1,1], [1,0], [0,1], [0,0]])
Y = np.array([0,1,1,0])

# Ref: https://github.com/erikdelange/Neural-networks-in-numpy/blob/master/xor.py
# Initial weights and bias
W1 = np.random.normal(size=(2, 2))  # layer 1 weights
W2 = np.random.normal(size=(1, 2))  # layer 2 weights
B1 = np.random.random(size=(2, 1))  # layer 1 bias
B2 = np.random.random(size=(1, 1))  # layer 2 bias

def nn(x):
    g1 = x[0].dot(x[4].T) + x[2]
    s = pow((1 + np.exp(-1 * g1)), -1.0)
    g2 = x[1].dot(s) + x[3]
    return g2

Xt  = ((1,1),(1,0),(0,1),(0,0))     # R^4x2
Xtt = ((1,1,0,0),(1,0,1,0))         # R^2x4 (transpose)
Yt  = ((0),(1),(1),(0))             # R^4x1
W1t = ((5,5), (5,5))  # R^2x2
W2t = ((3,3))         # R^1x2
B1t = ((4),(4))       # R^2x1
B2t = ((4))           # R^1x1



# R^2x2 * R^4x2 -> R^2x4
def dot1(MS):
    M1 = MS[0]
    m1t1 = M1[0]
    m1t2 = M1[1]
    M2 = MS[1]
    m2t1 = M2[0]
    m2t2 = M2[1]
    m2t3 = M2[2]
    m2t4 = M2[3]
    a1 = (m1t1[0]*m2t1[0] + m1t1[1]*m2t1[1])
    a2 = (m1t1[0]*m2t2[0] + m1t1[1]*m2t2[1])
    a3 = (m1t1[0]*m2t3[0] + m1t1[1]*m2t3[1])
    a4 = (m1t1[0]*m2t4[0] + m1t1[1]*m2t4[1])
    b1 = (m1t2[0]*m2t1[0] + m1t2[1]*m2t1[1])
    b2 = (m1t2[0]*m2t2[0] + m1t2[1]*m2t2[1])
    b3 = (m1t2[0]*m2t3[0] + m1t2[1]*m2t3[1])
    b4 = (m1t2[0]*m2t4[0] + m1t2[1]*m2t4[1])
    return ((a1,a2,a3,a4),(b1,b2,b3,b4))


def nnt(xs):
    g1 = xs[0].dot(x[4].T) + x[2]
    s = pow((1 + np.exp(-1 * g1)), -1.0)
    g2 = x[1].dot(s) + x[3]
    return g2



learning_rate = 1e-6
T = 1
# Gradient descend algorithm
for t in range(T):
    y_pred = nn([W1, W2, B1, B2, X])
    print(y_pred)

    # Compute and print loss
    loss = np.square(y_pred - Y).sum()
    if t % 100 == 99:
        print(t, loss)

    # compute gradients of weights and biases with respect to loss using adjoint CAD
    dw1, dw2, db1, db2, _, _, _ = lossdx.sqloss_diff_reverse(W1, W2, B1, B2, X, Y, 1.0)

    # Update weights
    W1 -= learning_rate * dw1
    B1 -= learning_rate * db1
    W2 -= learning_rate * dw2
    B2 -= learning_rate * db2

Y_hat = nn([W1, W2, B1, B2, X])

print("Predict: ", Y_hat)
print("Result: ", Y)