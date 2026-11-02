import numpy as np

def train_softmaxreg(X, y, learning_rate, iterations):
    n, d = X.shape
    C = np.max(y) + 1

    Xc = np.concatenate([np.ones((n, 1)), X], axis=1)
    W = np.zeros((d + 1, C))
    Y = np.eye(C)[y]

    losses = []

    for i in range(iterations):
        logits = Xc @ W
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if i == 0:
            losses.append(-np.sum(np.log(probs[np.arange(n), y])))

        grad = Xc.T @ (probs - Y)
        W -= learning_rate * grad

        if i < iterations - 1:
            logits = Xc @ W
            logits -= np.max(logits, axis=1, keepdims=True)

            log_probs = logits - np.log(
                np.sum(np.exp(logits), axis=1, keepdims=True)
            )
            losses.append(-np.sum(log_probs[np.arange(n), y]))

    return W.T.tolist(), losses