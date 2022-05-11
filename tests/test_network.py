from neural_network.scripts import network

import random
import numpy as np
import pandas as pd
from copy import deepcopy

import unittest


def get_test_data() -> (list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, int]]):
    """Example training data with
    https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep dataset"""

    data = pd.read_csv("..\\data\\SaYoPillow.csv")

    output_data = list(data.pop("sl"))
    input_data = list(data.to_numpy().reshape(630, 8, 1))
    training_outputs = []

    for i in output_data:
        d = np.zeros((5, 1))
        d[i, 0] = 1
        training_outputs.append(d)

    test_training_data = list(zip(input_data, training_outputs))[:int(len(input_data) * .9)]
    test_testing_data = list(zip(input_data, output_data))[int(len(input_data) * .9):]

    return test_training_data, test_testing_data


class TestNetwork(unittest.TestCase):

    def test_training_accuracy(self):
        training_data, testing_data = get_test_data()

        self.assertTrue(training_data and testing_data)

        net = network.NeuralNetwork(
            [training_data[0][0].size, 30, training_data[0][1].size]
        )

        net.train(training_data, epoch=30, mini_batch_size=20, test_data=testing_data)

        print(f"accuracy: {round(net.test(testing_data) / len(testing_data) * 100, 2)}%")

        # check if accuracy is better than random
        self.assertGreater((net.test(testing_data)/len(testing_data)), 1/training_data[0][1].size)

        # check if accuracy is above 60%
        self.assertGreater((net.test(testing_data)/len(testing_data)), .6)


    def test_backpropagation(self):
        net = network.NeuralNetwork([random.randint(1, 5) for _ in range(random.randint(3, 5))])

        inp_shape = net.shape[0]

        test_inp = np.array([[random.randint(0, 10)] for _ in range(inp_shape)])
        output = net.calculate(test_inp)

        layers_copy = deepcopy(net.layers)

        net.train([(test_inp, output)], mini_batch_size=1)

        # the weights and biases should not change if correct output is equal to output of the network
        self.assertTrue(all(
                all((  # check if weights and biases are same in layers_copy and net.layers
                    (layers_copy[layer].weights == net.layers[layer].weights).all(),
                    (layers_copy[layer].biases == net.layers[layer].biases).all()
                ))
                for layer in range(1, len(layers_copy))  # for each layer
        ))

        # proof that layers_copy and net.layers are not the same object
        self.assertIsNot(layers_copy, net.layers)
        self.assertTrue(all([layers_copy[i] is not net.layers[i] for i in range(len(layers_copy))]))