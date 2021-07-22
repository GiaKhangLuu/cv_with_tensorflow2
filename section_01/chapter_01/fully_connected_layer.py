import numpy as np


class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
    Args:
        num_inputs (int): The input vector size / number of input values.
        layer_size  (int): The output vector size / number of neurons.
        activation_fn (callable): The activation function for this layer.
    Attributes:
        W (ndarray): The weight values for each input.
        b (ndarray): The bias value, added to the weighted sum.
        size (int): The layer size / number of neurons.
        activation_fn (callable): The neuron's activation fn.
    """

    def __init__(self, num_inputs, layer_size, activation_fn):
        super().__init__()
        # Randomly initializing the parameters (using a normal distribution this time).
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self, x):
        """Forward the input signal through the layer."""
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)


np.random.seed(42)

# Random input column-vectors of 2 values (shape = `(1, 2)`):
x1 = np.random.uniform(low=-1, high=1, size=2).reshape(1, 2)
x2 = np.random.uniform(low=-1, high=1, size=2).reshape(1, 2)

# Define relu function
relu_fn = lambda y: np.maximum(0, y)

# Instantiating a FC layer
layer = FullyConnectedLayer(x1.size, 3, relu_fn)

# Our layer can process x1 and x2 separately ...
out1 = layer.forward(x1)
print("out1=", out1)

out2 = layer.forward(x2)
print("out2=", out2)

# ... or together:
x12 = np.concatenate((x1, x2))
out12 = layer.forward(x12)
print("out12=", out12)