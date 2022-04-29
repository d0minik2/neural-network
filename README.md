# Neural Network

<hr>

My first attempt to create neural networks from scratch.


## Example usage

<hr>

Example usage with [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/). The network will predict what this number is based on the image.

### Importing neural network model and MNIST dataset

```python
import mnist
from network import NeuralNetwork, load_network
```


### Getting MNIST dataset

```python
training_data, test_data = mnist.get_data()
```


### Creating a model with 784 input neurons, 100 and 70 hidden layer neurons, and 10 output neurons. Then training the model using the MNIST dataset with 15 epochs and a mini batch size of 20.

```python
network = NeuralNetwork([784, 100, 70, 10])
network.train(training_data, epoch=15, mini_batch_size=20, test_data=test_data)
```
_epoch 1/15 | 1077/10000_

_..._

_epoch 15/15 | 8949/10000_


### Or loading it from file

```python
network = load_network("data\\network.pickle")
```


### Checking accuracy of the network

```python
print(f"Accuracy is {round(network.test(test_data) / len(test_data) * 100, 1)}%")
```

_Accuracy is 89.5%_


### Saving network a file

```python
network.save("data\\network.pickle")
```