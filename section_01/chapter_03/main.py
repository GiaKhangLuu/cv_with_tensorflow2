import numpy as np
from simple_conv_layer import SimpleConvolutionLayer


conv2d = SimpleConvolutionLayer()

input = np.random.rand(1, 256, 256, 3)
conv2d.build(input)
print(conv2d.kernels.shape)
print(conv2d.bias.shape)
print((conv2d.kernels + conv2d.bias).shape)