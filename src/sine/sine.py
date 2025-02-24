import subprocess
import numpy as np
import math
import loss
import lossdx
import matplotlib.pyplot as plt

def sin():
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
sin()

# Reference: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
def sin2():
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    # Gradient descend algorithm
    for t in range(2000):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        X = [a, b, c, d, x]
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            print(t, loss)

        # compute gradients of a, b, c, d with respect to loss using adjoint CAD
        grad_a, grad_b, grad_c, grad_d, _ = lossdx.l_diff_reverse(a,b,c,d,x,1.0)

        # Update weights
        a -= learning_rate * grad_a.sum()
        b -= learning_rate * grad_b.sum()
        c -= learning_rate * grad_c.sum()
        d -= learning_rate * grad_d.sum()

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Sine")
    ax.plot(x, y_pred, label="Polynomial approximation")

    ax.legend()

    ax.set(xlabel='x', ylabel='y',
        title='Approximating Sine Function with Gradient Descent')
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
sin2()