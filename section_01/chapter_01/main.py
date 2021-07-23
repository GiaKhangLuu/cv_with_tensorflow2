from keras.datasets import mnist
import numpy as np
import ssl
import requests

from ann import SimpleNetwork

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

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

losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test, batch_size=1)

