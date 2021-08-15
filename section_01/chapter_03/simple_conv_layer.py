import numpy as np
import tensorflow as tf


class SimpleConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, num_kernels=32, kernel_size=(3, 3), strides=1):
        """Initialize convolutional layer.
        Args:
            num_kernels (int): number of kernels.
            kernel_size (tuple): kernel size (H x W).
            strides (int): verical/horizontal stride.
        """

        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        """Build the layer, initializing its parameters/variables.
        This will be the internally called the 1st time the layer is used.
        Args:
            input_shape (ndarray): Input shape for the layer (for instance, B x H x W x C).
        """

        num_input_ch = input_shape.shape[-1]
        # Shape of the kernal [k_H, k_W, D, N]
        kernels_shape = (*self.kernel_size, num_input_ch, self.num_kernels)
        # We initialize the filter values for instace, from a Glorot distributions:
        glorot_init = tf.initializers.GlorotUniform()
        # Method add Variables to layer and we make it trainable.
        self.kernels = self.add_weight(name='kernels', shape=kernels_shape,
                                       initializer=glorot_init, trainable=True)
        # Same for the bias variable (for instance, from a normal distribution):
        self.bias = self.add_weight(name='bias', shape=(self.num_kernels,),
                                    initializer='random_normal', trainable=True)

    def call(self, inputs):
        """Call the layer and apply its operations to the input tensor.
        Args:
            inputs (ndarray): input tensor.
        :return: output tensor.
        """

        # We perform the convolution
        z = tf.nn.conv2d(input, self.kernels, strides=[1, *self.strides, 1], padding='VALID')
        if self.use_bias:
            z += self.bias
        # Finally, we apply the activation function
        return tf.nn.relu(z)
