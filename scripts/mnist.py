import pickle
import gzip
import os
import numpy as np


def get_data() -> (list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, int]]):
    """Returns the MNIST data"""

    with gzip.open('data\\mnist.pkl.gz', 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    training_d, _, testing_d = data

    training_data = list(zip(
        [np.reshape(inp, (784, 1)) for inp in training_d[0]],  # training inputs
        [np.reshape(np.identity(10)[out], (10, 1)) for out in training_d[1]]  # training outputs
    ))

    testing_data = list(zip(
        [np.reshape(x, (784, 1)) for x in testing_d[0]],  # testing inputs
        testing_d[1]  # testing outputs (correct number, not a ndarray)
    ))

    return training_data, testing_data
