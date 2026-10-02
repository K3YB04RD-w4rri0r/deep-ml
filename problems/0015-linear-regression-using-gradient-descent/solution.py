import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape # row form
    y = y.reshape(-1, 1)  # Ensure y is a column vector # m points
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    for epochs in range(iterations):
        # optim.zero_grad()
        # forward
        y_hat = X @ theta # shape m, 1
        loss = (1/2) * np.mean((y_hat - y)**2)
        # backwards
        # dL/dw = X @ ((y_hat - y) / len(y))
        g = X.T @ ((y_hat - y) / len(y))
        theta = theta - alpha * g


    return theta.flatten()