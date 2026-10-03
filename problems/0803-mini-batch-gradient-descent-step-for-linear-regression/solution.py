import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    n, d = X.shape # weights (d,)
    m = len(batch_indices)
    # single step
    y_hat = X[batch_indices] @ weights + bias 
    # loss = np.sum((y_hat - y[batch_indices])**2)/m
    dW  = (1/m) * 2 * X[batch_indices].T @ (y_hat - y[batch_indices]) # (d, m)(m,) -> (d,)
    db  = (1/m) * 2 * np.sum(y_hat - y[batch_indices]) #

    w = weights - lr * dW
    b = bias - lr * db

    return np.concatenate([w, np.array([b])])





