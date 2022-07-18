from src import network

import random
import numpy as np
import pandas as pd

import unittest


class TestEquations(unittest.TestCase):
    h = 1e-5

    def test_errorOut(self):
        out = random.randint(1, 10)
        z = random.randint(1, 5) * random.randint(1, 10) + random.randint(1, 5)
        cost = network.cost(network.sigmoid(z), out)

        error_out_test = (network.cost(network.sigmoid(z+self.h), out) - cost) / self.h * cost

        error_out = network.errorOut(cost, network.sigmoid(z))

        self.assertAlmostEqual(error_out, error_out_test, delta=self.h*1e2)

    def test_errorL(self):
        out = random.randint(1, 10)
        z0 = network.sigmoid(random.randint(1, 5)) * random.randint(1, 10) + random.randint(1, 5)  # l-1

        # int has no attribute transpose
        weights = type("int", (int,), {"transpose": lambda self: self})(random.randint(1, 10))
        bias = random.randint(1, 5)
        z = network.sigmoid(z0) * weights + bias

        error_out = network.errorOut(
                    network.cost(network.sigmoid(z), out),
                    network.sigmoid(z)
        )

        error_l_test = -(
                network.errorOut(
                    network.cost(network.sigmoid(network.sigmoid(z0 + self.h) * weights + bias), out),
                    network.sigmoid(network.sigmoid(z0 + self.h) * weights + bias)
                ) - error_out
        ) / self.h

        error_l = network.errorL(
            weights, error_out, network.sigmoid(z0)
        )

        self.assertAlmostEqual(error_l, error_l_test, delta=self.h*1e2)

    def test_s_weights(self):
        out = random.randint(1, 10)
        weights = random.randint(1, 10)
        bias = random.randint(1, 5)
        activations = network.sigmoid(random.randint(1, 5))
        z = activations * weights + bias

        error = network.errorOut(network.cost(activations, out), network.sigmoid(z))

        d_z_weights = ((activations * (weights + self.h) + bias) - z) / self.h
        s_weights_test = d_z_weights * error

        s_weights = network.s_weights(activations, error)

        self.assertAlmostEqual(s_weights, s_weights_test, delta=self.h*1e2)

    def test_s_biases(self):
        out = random.randint(1, 10)
        weights = random.randint(1, 10)
        bias = random.randint(1, 5)
        z = random.randint(1, 5) * weights + random.randint(1, 5)
        activations = network.sigmoid(random.randint(1, 5))

        error = network.errorOut(network.cost(activations, out), network.sigmoid(z))

        d_z_biases = ((activations * weights + (bias + self.h)) - (activations * weights + bias)) / self.h
        s_biases_test = d_z_biases * error

        s_biases = network.s_biases(error)

        self.assertAlmostEqual(s_biases, s_biases_test, delta=self.h*1e2)
