import random
import pickle
import numpy as np



class InputLayer:
    """Neural network input layer"""

    def __init__(
            self,
            size: int,
    ):
        self.size = size

    def activate(
            self,
            inputs: np.ndarray,
    ) -> np.ndarray:
        """Activates input neurons"""

        assert inputs.shape == (self.size, 1)
        assert np.all(inputs is not np.nan)

        return inputs


class Layer:
    """Neural Network Layer"""

    def __init__(
            self,
            size: int,
            previous_size: int
    ):
        self.size = [previous_size, size]
        self.weights = np.random.randn(size, previous_size)
        self.biases = np.random.randn(size, 1)

    def activate(
            self,
            previous_activations: np.ndarray,
            return_z=False
    ) -> np.ndarray or (np.ndarray, np.ndarray):
        """Activates this layer"""

        assert previous_activations.shape == (self.size[0], 1)
        assert np.all(self.weights is not np.nan)
        assert np.all(self.biases is not np.nan)

        z = np.dot(self.weights, previous_activations) + self.biases
        activations = sigmoid(z)

        if return_z is True:
            return activations, z

        return activations


class BackpropLayer:
    """Temporary layer used for backpropagation"""

    def __init__(
            self,
            layer: Layer,
    ):
        self.layer = layer
        self.mbs = 1  # (needed to count the mean)

        if not isinstance(self.layer, InputLayer):
            self.weights = self.layer.weights
            self.biases = self.layer.biases

            self.ratio_biases = np.zeros(self.biases.shape)
            self.ratio_weights = np.zeros(self.weights.shape)

            self.z = None

        self.activation = None

    def activate(self, previous_activations):
        """Activates layer"""

        if not isinstance(self.layer, InputLayer):
            self.activation, self.z = self.layer.activate(
                previous_activations,
                return_z=True
            )

        else:
            self.activation = self.layer.activate(previous_activations)

        return self.activation

    def train(
            self,
            ratio_weights: np.ndarray,
            ratio_biases: np.ndarray
    ):
        """"""

        self.ratio_weights += ratio_weights
        self.ratio_biases += ratio_biases
        self.mbs += 1

    def update_layer(self):
        """Updates weights and biases for that layer"""

        if not isinstance(self.layer, InputLayer):
            self.layer.weights = (self.weights - (self.ratio_weights / self.mbs))
            self.layer.biases = (self.biases - (self.ratio_biases / self.mbs))

            self.weights = self.layer.weights
            self.biases = self.layer.biases

        return self.layer


class NeuralNetwork:

    def __init__(
            self,
            shape: list[int],
    ):
        """shape: list of sizes of all layers"""

        self.shape = shape
        self.layers: list[InputLayer or Layer] = []
        self.backprop_layers: list[BackpropLayer] = []

        self.__generate_layers()

    def __str__(self):
        return str("\n".join([str(layer.weights) for layer in self.layers[1:]]))

    def __generate_layers(self):
        """Generates the layers"""

        self.layers.append(
            InputLayer(self.shape[0])
        )

        for index, size in enumerate(self.shape[1:]):
            self.layers.append(
                Layer(size, self.shape[index])
            )

    def __generate_backprop_layers(self):
        """Generates temp layers"""

        self.backprop_layers = [
            BackpropLayer(layer)
            for layer in self.layers
        ]

    def calculate(self, inputs: np.ndarray, argmax=False) -> np.ndarray:
        """Returns the result of the network"""

        activation = inputs

        for layer in self.layers:
            activation = layer.activate(activation)

        if argmax:
            return np.argmax(activation)

        return activation

    def backpropanagtion(self, inp: np.ndarray, correct_output: np.ndarray):
        """Backpropagation"""

        activation = inp  # current activation

        # activate backprop layers
        for layer in self.backprop_layers:
            activation = layer.activate(activation)

        # error = (∂activations[l] / ∂z[l]) ⊙ (∂Cost / ∂activations[l])
        error = errorOut(
            cost(
                activation,
                correct_output
            ), sigmoid(self.backprop_layers[-1].z)
        )

        self.backprop_layers[-1].train(
            s_weights(self.backprop_layers[-2].activation, error),
            s_biases(error)
        )

        # computes the ratio for every layer starts at -2
        for layer in range(2, len(self.backprop_layers)-1):

            error = errorL(self.backprop_layers[-(layer - 1)].weights, error, sigmoid(self.backprop_layers[-layer].z))

            self.backprop_layers[-layer].train(
                s_weights(self.backprop_layers[-(layer + 1)].activation, error),
                s_biases(error)
            )

    def train(
            self,
            training_data: list[tuple[np.ndarray, np.ndarray]],
            epoch=1,
            mini_batch_size=10,
            test_data=None
    ) -> None:
        """Trains the naural network using stochastic gradient descent"""

        for j in range(epoch):  # repeat training process epoch times
            print(f"\nepoch {j+1}/{epoch} | " + str(str(self.test(test_data)) + "/" +
                                                    str(len(test_data)) if test_data is not None else ""))

            random.shuffle(training_data)

            for i in range(0, len(training_data), mini_batch_size):  # split on mini batches
                self.__generate_backprop_layers()  # reset backprop layers

                mini_batch = training_data[i:i+mini_batch_size]  # get mini batch

                for x, y in mini_batch:
                    self.backpropanagtion(x, y)

                # update all layers
                for layer in range(len(self.layers)):
                    self.layers[layer] = self.backprop_layers[layer].update_layer()

    def test(self, test_data) -> int:
        """Returns the count of correct calculated data"""

        return sum(np.argmax(self.calculate(x)) == y for x, y in test_data)

    def save(self, path):
        """Saves network to file"""

        with open(path, "wb") as file:
            pickle.dump(self, file)


def load_network(path) -> NeuralNetwork or False:
    """Returns network loaded from file"""
    try:
        with open(path, "rb") as file:
            net = pickle.load(file)
    except FileNotFoundError:
        return False

    return net


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function σ(z) = (1 / (1 + e^-z))"""

    return 1 / (1 + np.exp(-z))


def cost(activations: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Calculates the cost for output activations"""

    return activations - out


#### equations of backpropagation

def errorOut(cost_a: np.ndarray, sigm: np.ndarray) -> np.ndarray:  # BP1
    """Error for last layer: errorOut = (∂activations[l] / ∂z[l]) * (∂Cost / ∂activations[l])
    = 2*cost(activations[-1]) ⊙ sigmoid`(z[-1])
    sigmoid` gets small if the sigmoid is near saturation
    """

    return 2*cost_a * (sigm * (1 - sigm))


def errorL(weights: np.ndarray, error: np.ndarray, sigm: np.ndarray) -> np.ndarray:  # BP2
    """Error for current layer: errorL = (∂activations[l] / ∂z[l]) * (∂Cost / ∂activations[l])
    = 2*((weights[l+1]).transpose() * error[l+1]) ⊙ sigmoid(z[l])"""

    return 2*np.dot(weights.transpose(), error) * (sigm*(1-sigm))


def s_biases(error: np.ndarray) -> np.ndarray:  # BP3
    """Sensivity error to biases: ∂Cost / ∂biases = error"""

    return error


def s_weights(activations: np.ndarray, error: np.ndarray) -> np.ndarray:  # BP4
    """Sensivity error to weights: ∂Cost / ∂weights = activations[l-1].transpose() * error[l]"""

    return np.dot(error, activations.transpose())
