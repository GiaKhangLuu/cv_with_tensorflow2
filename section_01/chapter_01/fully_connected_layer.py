import numpy as np


class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
    Args:
        num_inputs (int): The input vector size / number of input values.
        layer_size  (int): The output vector size / number of neurons.
        activation_fn (callable): The activation function for this layer.
        derivated_activation_fn (callable): The derivated activation function for this layer.
    Attributes:
        W (ndarray): The weight values for each input.
        b (ndarray): The bias value, added to the weighted sum.
        size (int): The layer size / number of neurons.
        activation_fn (callable): The neuron's activation fn.
        x (ndarray): The last provided input vector, stored for backpropagation.
        y (ndarray): The corresponding output, also stored for backpropagation.
        derivated_activation_fn (callable): The corresponding derivated function for backpropagation.
        dL_dW (ndarray): The derivative of the loss, with respect to the weights W.
        dL_db (ndarray): The derivative of the loss, with respect to the bias b.
    """

    def __init__(self, num_inputs, layer_size, activation_fn, derivated_activation_fn=None):
        super().__init__()

        # Randomly initializing the parameters (using a normal distribution this time).
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn
        self.derivated_activation_fn = derivated_activation_fn
        self.x, self.y, self.dL_dW, self.dL_db = None, None, None, None

    def forward(self, x):
        """Forward the input signal through the layer."""
        z = np.dot(x, self.W) + self.b
        self.x = x  # we stored the input and the ouput values for back-prop
        self.y = self.activation_fn(z)
        return self.y

    def backward(self, dL_dy):
        """
        Backpropagate the loss, computing all the derivatives, storing those w.r.t. the
        layer parameters, and returning the loss w.r.t. its input for further propagation.
        :param
            dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
        :return:
            dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).
        """
        dy_dz = self.derivated_activation_fn(self.y)  # f'
        dL_dz = dL_dy * dy_dz  # dL/dz = dL/dy * dy/dz = l'_{k+1} * f'
        dz_dw = self.x.T
        dz_db = np.ones(dL_dy.shape[0])
        dz_dx = self.W.T

        # Computing the derivatives w.r.t. the layer's parameters and storing them for opt.
        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        # Computing the derivaties w.r.t. the input, to be passed to the previous layers (their `dL_dy`).
        dl_dx = np.dot(dL_dz, dz_dx)
        return dl_dx

    def optimize(self, epsilon):
        """
        Opt. the layer's parameters, using store derivative values.
        :param
            epsilon (float): The learning rate.
        :return:
        """
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db


# ========================================================
# Main call
# ========================================================

"""
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
"""
