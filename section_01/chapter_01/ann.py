import numpy as np

from fully_connected_layer import FullyConnectedLayer


# ===========================================
# Function definitions
# ===========================================

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define sigmoid derivative function
def derivated_sigmoid(y):
    return y * (1 - y)


# L2 loss function
def loss_L2(pred, target):
    return np.sum(np.square(pred - target)) / pred.shape[0]


# L2 derivative function
def derivated_loss_L2(pred, target):
    return 2 * (pred - target)


# Cross entropy function
def cross_entropy(pred, target):
    return -np.mean(np.multiply(np.log(pred), target) + np.multiply(np.log(1 - pred), (1 - target)))


# Cross entropy derivative function
def derivated_cross_entropy(pred, target):
    return (pred - target) / (pred * (1 - pred))


# ==========================================
# Class definitions
# ==========================================

class SimpleNetwork(object):
    """A simple fully-connected NN.
    Args:
         num_inputs (int): The input vector size / number of input values.
         num_outputs (int): The output vector size.
         hidden_layers_sizes (list): A list of sizes for each hidden layer to be added to the network.
         activation_function (callable): The activation function for all the layers.
         derivated_activation_function (callable): The derivated activation function.
         loss_function (callable): The loss function to train this network.
         derivated_loss_function (callable): The derivative of the loss function, for back-prop.
    Attributes:
        layers (list): The list of layers forming this simple network.
        loss_function (callable): The loss function to train this network.
        derivated_loss_function (callable): The derivative of the loss function, for back-prop.
    """

    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes,
                 activation_function=sigmoid, derivated_activation_function=derivated_sigmoid,
                 loss_function=loss_L2, derivated_loss_function=derivated_loss_L2):
        super().__init__()

        # We build the list of layers composing the network:
        sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [FullyConnectedLayer(sizes[i], sizes[i + 1], activation_function, derivated_activation_function)
                       for i in range(len(sizes) - 1)]
        self.loss_function = loss_function
        self.derivated_loss_function = derivated_loss_function

    def forward(self, x):
        """Forward the input vector `x` through the layers."""
        for layer in self.layers:  # from the input layer to the output one
            x = layer.forward(x)
        return x

    def backward(self, dL_dy):
        """
        Back-propagation the loss through the layers (require `forward()` to be called before).
        :param
            dL_dy (ndarray): The loss derivative w.r.t. the network's output (dL/dy).
        :return:
            dL_dx (ndarray): The loss derivative w.r.t. the network's input (dL/dx).
        """
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon):
        """
        Optimize the network's parameters according to the stored gradients (require `backward()` to be called before).
        :param
            epsilon (float): The learning rate
        :return:
        """
        for layer in self.layers:  # the order doesnt matter here
            layer.optimize(epsilon)

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

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=16, num_epochs=5, learning_rate=1e-3):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.
        :param X_train (ndarray): The input training dataset.
        :param y_train (ndarray): The corresponding ground-truth training dataset.
        :param X_val (ndarray): The input validation dataset.
        :param y_val (ndarray): The corresponding groung-truth validation dataset.
        :param batch_size (int): The mini-batch size.
        :param num_epochs (int): The number of training epochs i.e. iterations over the whole dataset.
        :param learning_rate (float): The learning rate to scale the derivatives
        :return losses (list): The list of training losses for each epoch.
        :return accuracies (list): The list of training accuracy for each epoch.
        """
        num_batches_per_epoch = X_train.shape[0] // batch_size
        do_validation = X_val is not None and y_val is not None
        losses, accuracies = [], []
        for i in range(num_epochs):  # for each training epoch
            epoch_loss = 0
            for b in range(batch_size):  # for each batch composing the dataset:
                batch_index_begin = b * batch_size
                batch_index_end = batch_index_begin + batch_size
                x = X_train[batch_index_begin: batch_index_end]
                targets = y_train[batch_index_begin: batch_index_end]
                # Optimize on batch
                predictions = self.forward(x)  # forward pass
                L = self.loss_function(predictions, targets)  # loss computation
                dL_dy = self.derivated_loss_function(predictions, targets)  # loss derivation
                self.backward(dL_dy)  # back-propagation
                self.optimize(learning_rate)  # optimization of the NNs
                epoch_loss += L

            # Logging training loss and validation accuracy, to follow the training:
            epoch_loss /= num_batches_per_epoch
            losses.append(epoch_loss)
            if do_validation:
                accuracy = self.evaluate_accuracy(X_val, y_val)
                accuracies.append(accuracy)
            else:
                accuracies.append(np.NAN)
            print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%"
                  .format(i, epoch_loss, accuracy * 100))
        return losses, accuracies
