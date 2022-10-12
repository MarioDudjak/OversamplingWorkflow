import numpy as np


def preprocess(dataset):
    data = np.array(dataset['data'])
    data = np.unique(data, axis=0)
    X = data[:, :-1]
    y = data[:, -1]

    X = X.astype(np.float64)
    y = y.astype(np.uint32)

    return X, y
