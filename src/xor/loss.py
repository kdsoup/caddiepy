# first layer   R^2 -> R^2
# sigmoid       R^2 -> R^2
# second layer  R^2 -> R

def dot(xs): return xs[0] * xs[1]

def sigmoid(x): return pow((1 + exp(-1 * x[1])), -1.0)

# squared loss (y - y')^2
def sqloss(x): w1 = x[0]; w2 = x[1]; b1 = x[2];  b2 = x[3]; x1 = x[4]; y = x[5]; return pow(((w2 * sigmoid((w1 * x1 + b1)) + b2)- y), 2.0)
