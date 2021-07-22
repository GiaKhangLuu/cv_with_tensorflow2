import numpy as np
from keras.datasets import mnist
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

from fully_connected_layer import FullyConnectedLayer


# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SimpleNetwork(object):
    """A simple fully-connected NN.
    Args:
         num_inputs (int): The input vector size / number of input values.
         num_outputs (int): The output vector size.
         hidden_layers_sizes (list): A list of sizes for each hidden layer to be added to the network.
    Attributes:
        layers (list): The list of layers forming this simple network.
    """

    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes):
        super().__init__()
        # We build the list of layers composing the network:
        sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [FullyConnectedLayer(sizes[i], sizes[i + 1], sigmoid) for i in range(len(sizes) - 1)]

    def forward(self, x):
        """Forward the input vector `x` through the layers."""
        for layer in self.layers:  # from the input layer to the output one
            x = layer.forward(x)
        return x

    def predict(self, x):
        """Compute the output corresponding to `x` and return the index of the largest output value."""
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def evaluate_accuracy(self, X_val, y_val):
        """Evaluate the network's accuracy on validation dataset."""
        num_correct = 0
        for i in range(len(X_val)):
            if self.predict(X_val[i]) == y_val[i]:
                num_correct += 1
        return num_correct / len(X_val)


np.random.seed(42)

# Loading the training and testing data:
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Classes are the digits from 0 to 9
num_classes = 10

# We transform the images into columns vectors (as inputs for out NN):
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

# We "one-hot" the labels (as targets for our NN), for instance,
# transform label `4` into vector `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]`:
y_train = np.eye(num_classes)[y_train]

# Network for MNIST images, with 2 hidden layers of size 64 and 32:
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, (64, 32))

# ... and we evaluate its accuracy on the MNIST test set:
accuracy = mnist_classifier.evaluate_accuracy(X_test, y_test)
print("Accuracy = {:.2f}%".format(accuracy * 100))