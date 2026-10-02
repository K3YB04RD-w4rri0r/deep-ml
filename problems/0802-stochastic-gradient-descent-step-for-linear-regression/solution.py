import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n, d = X.shape
    # print(weights.shape)
    for epoch in range(n_iter):
        i = epoch % n
        # forward for one sample
        yihat = X[i] @ weights # 1,d,d,1 -> 1,1
        g = 2 * X[i].T * (yihat - y[i]) # d,1,1,1 -> d,1
        weights -= learning_rate * g
    return weights.tolist()

