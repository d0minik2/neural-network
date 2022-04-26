import mnist
from network import NeuralNetwork


def main():
    training_data, test_data = mnist.get_data()

    network = NeuralNetwork([784, 70, 40, 10])
    network.train(training_data, epoch=15, mini_batch_size=20, test_data=test_data)


    # net = load_network("data\\network.pickle")
    network.save("data\\network.pickle")

    # _, test_data = mnist.get_data()
    print(f"{network.test(test_data) / len(test_data) * 100}%")

    # t = test_data[int(input())]
    # print(f"result={network.calculate(t[0], argmax=True)}, correct={t[1]}")


if __name__ == "__main__":
    main()