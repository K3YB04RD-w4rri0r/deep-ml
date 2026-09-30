import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = len(data)
    indexes  = np.random.default_rng(seed).permutation(n)
    # print(indexes)
    data  = data[indexes]
    train = data[:int(n*train_frac)]
    val = data[int(n*train_frac): int(n*train_frac) + int(n*validation_frac)]
    test = data[int(n*train_frac) + int(n*validation_frac):]
    return [train, val, test]