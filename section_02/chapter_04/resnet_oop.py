# ===================================================================
# Imported Modules
# ===================================================================
from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, Activation,
                                     add, GlobalAvgPool2D, Dense, Layer, InputLayer)
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.regularizers as regularizers
from functools import partial


# ===================================================================
# Class Definitions
# ===================================================================


class ConvWithBatchNorm(Conv2D):
    """Conv layer with batch norm"""

    def __init__(self, activation='relu', name='convb', **kwargs):
        """
        Initialize the layer
        :param activation: Activation function (name or callable)
        :param name: Name suffix for the sub-layers
        :param kwargs: Optional parameters of Conv2D
        """

        super().__init__(activation=None, name=name + '_c', **kwargs)
        self.activation = (Activation(activation, name=name + '_act')
                           if activation is not None else None)
        self.batch_norm = BatchNormalization(name=name + '_b')

    def call(self, inputs):
        """
        Call the layer
        :param inputs: Input tensor
        :return: Convolved tensor
        """

        x = super().call(inputs)
        x = self.batch_norm(x)
        x = self.activation(x) if self.activation is not None else x

        return x


class ResidualMerge(Layer):
    """Layer to merge the original tensor and the residual one in residual block"""

    def __init__(self, name, **kwargs):
        """
        Initializer the layer
        :param name: Name suffix for the sub-layers
        :param kwargs: Optional parameters of Conv2D
        """

        super().__init__(name=name)
        self.shortcut = None
        self.kwargs = kwargs

    def build(self, input_shape):
        """
        Build the layer
        :param input_shape: Tuple of input shape
        :return:
        """

        x_shape = input_shape[0]
        x_residual_shape = input_shape[1]
        if (x_shape[1] == x_residual_shape[1] and
                x_shape[2] == x_residual_shape[2] and
                x_shape[3] == x_residual_shape[3]):
            self.shortcut = partial(tf.identity, name=self.name + '_shortcut')
        else:
            strides = (int(round(x_shape[2] / x_residual_shape[2])),  # horizontal strides
                       int(round(x_shape[1] / x_residual_shape[1])))  # vertical strides
            num_channels = x_residual_shape[3]
            self.shortcut = ConvWithBatchNorm(filters=num_channels, kernel_size=1, strides=strides,
                                              activation=None, name=self.name + '_shortcut', **self.kwargs)

    def call(self, inputs):
        """
        Call the layer
        :param inputs: Input tensor
        :return x_merge: Merged tensor
        """

        x, x_residual = inputs
        x_shortcut = self.shortcut(x)
        x_merge = add([x_shortcut, x_residual])

        return x_merge


class BasicResidualBlock(Model):
    """Basic residual block"""

    def __init__(self, filters=64, kernel_size=3, strides=1, activation='relu', name='res_basic',
                 kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), **kwargs):
        """
        Initializer the block
        :param filters: Number of filters
        :param kernel_size: Kernel size
        :param strides: Convolution strides
        :param activation: Activation function (name or callable)
        :param name: Name suffix for the sub_layers
        :param kernel_initializer: Kernel initializer method name
        :param kernel_regularizer: Kernel regularizer
        :param kwargs: Optional parameters of Conv2D
        """

        super().__init__(name=name)

        self.conv_01 = ConvWithBatchNorm(filters=filters, kernel_size=kernel_size, strides=strides,
                                         padding='same', activation=activation, name=name + '_cb_01',
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, **kwargs)

        self.conv_02 = ConvWithBatchNorm(filters=filters, kernel_size=kernel_size, strides=1,
                                         padding='same', activation=None, name=name + '_cb_02',
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, **kwargs)

        self.merge = ResidualMerge(name=name + '_merge', kernel_regularizer=kernel_regularizer,
                                   kernel_initializer=kernel_initializer)

        self.activation = Activation(activation, name=name + '_act')

    def call(self, inputs):
        """
        Call the layer
        :param inputs:  Input tensor
        :return x_merge: Block ouput tensor
        """

        x = inputs

        # Residual path
        x_residual = self.conv_01(x)
        x_residual = self.conv_02(x_residual)

        # Merge path
        x_merge = self.merge([x, x_residual])
        x_merge = self.activation(x_merge)

        return x_merge


class ResidualBlockWithBottleNeck(Model):
    """Residual block with bottleneck, recommended for deep ResNet (layer > 34)"""

    def __init__(self, filters, kernel_size, strides, activation='relu',
                 name='res_bottleneck', kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4), **kwargs):
        """
        Initialize the layer
        :param filters: Number of filters
        :param kernel_size: Kernel size
        :param strides: Convolution strides
        :param activation: Activation function (name or callable)
        :param name: Name suffix for the sub-layers
        :param kernel_initializer: Kernel initializer method name
        :param kernel_regularizer: Kernel regularizer
        :param kwargs: Optional parameters for Conv2D
        """

        super().__init__(name=name)

        self.conv_01 = ConvWithBatchNorm(filters=filters, kernel_size=1, strides=strides,
                                         padding='valid', activation=activation, name=name + '_cb_01',
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, **kwargs)

        self.conv_02 = ConvWithBatchNorm(filters=filters, kernel_size=kernel_size, strides=1,
                                         padding='same', activation=activation, name=name + '_cb_02',
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, **kwargs)

        self.conv_03 = ConvWithBatchNorm(filters=filters * 4, kernel_size=1, strides=1,
                                         padding='valid', activation=None, name=name + '_cb_03',
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, **kwargs)

        self.merge = ResidualMerge(name=name + '_merge', kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer)

        self.activation = Activation(activation, name=name + '_act')

    def call(self, inputs):
        """
        Call the layer
        :param inputs: Input tensor
        :return x_merge: Block output tensor
        """

        x = inputs

        # Residual path
        x_residual = self.conv_01(x)
        x_residual = self.conv_02(x_residual)
        x_residual = self.conv_03(x_residual)

        # Merge path
        x_merge = self.merge([x, x_residual])
        x_merge = self.activation(x_merge)

        return x_merge


class MacroBlock(Sequential):
    """Macro_block, chaining multiple residual blocks (as a sequential model)"""

    def __init__(self, block_class=BasicResidualBlock, repetitions=2, filters=64,
                 kernel_size=3, strides=1, activation='relu', name='res_macroblock',
                 kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4),
                 **kwargs):
        """
        Initializer the layer
        :param block_class: Block class to be used
        :param repetitions: Number of times the block should be repeated inside
        :param filters: Number of filters
        :param kernel_size: Kernel size
        :param strides: Convolution strides
        :param activation: Activation function (name or callable)
        :param name: Name suffix for the sub_layers
        :param kernel_initializer: Kernel initializer method name
        :param kernel_regularizer: Kenrel regularizer
        :param kwargs: Optional parameters of Conv2D
        """

        model = [block_class(filters=filters, kernel_size=kernel_size,
                             strides=strides if i == 0 else 1, activation=activation,
                             name='{}_{}'.format(name, i),
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer, **kwargs)
                 for i in range(repetitions)]

        super().__init__(model, name=name)


class ResNet(Sequential):
    """ResNet model for classification"""

    def __init__(self, input_shape, num_classes=1000,
                 block_class=BasicResidualBlock, repetitions=(2, 2, 2, 2),
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4), name='resnet'):
        """
        Initialize a ResNet model for classification
        :param input_shape: Input shape
        :param num_classes: Number of classes
        :param block_class: Block class to be used
        :param repetitions: List of repetitions for each macro_block the network
                            should contain
        :param kernel_initializer: Kernel initializer method name
        :param kernel_regularizer: Kernel regularizer
        :param name: Model's name
        """

        filters, strides = 64, 2

        # Init conv and max pooling
        conv_01 = [InputLayer(input_shape=input_shape),
                   ConvWithBatchNorm(filters=filters, kernel_size=7, strides=strides,
                                     padding='same', activation='relu', name='conv',
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer),
                   MaxPooling2D(pool_size=3, strides=2, padding='same', name='max_pooling')]

        # Residual blocks
        res_blocks = [MacroBlock(block_class, repetitions=repet,
                                 filters=min(filters * (2 ** i), 1024), kernel_size=3,
                                 strides=1 if i == 0 else strides, activation='relu',
                                 name='block_{}'.format(i + 1),
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
                      for i, repet in enumerate(repetitions)]

        # Final layer
        predict_layer = [GlobalAvgPool2D(name='avg_pool'),
                         Dense(units=num_classes, activation='softmax', name=name + '_softmax')]

        super().__init__(conv_01 + res_blocks + predict_layer, name=name)


# Standard ResNet versions
class ResNet18(ResNet):

    def __init__(self, input_shape, num_classes=1000, name='resnet18',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes, name=name,
                         block_class=BasicResidualBlock, repetitions=(2, 2, 2, 2),
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)


class ResNet34(ResNet):

    def __init__(self, input_shape, num_classes=1000, name='resnet34',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes, name=name,
                         block_class=BasicResidualBlock, repetitions=(3, 4, 6, 3),
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)


class ResNet50(ResNet):

    def __init__(self, input_shape, num_classes=1000, name='resnet50',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes, name=name,
                         block_class=ResidualBlockWithBottleNeck, repetitions=(3, 4, 6, 3),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer)


class ResNet101(ResNet):

    def __init__(self, input_shape, num_classes=1000, name='resnet101',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes, name=name,
                         block_class=ResidualBlockWithBottleNeck, repetitions=(3, 4, 23, 3),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer)


class ResNet152(ResNet):

    def __init__(self, input_shape, num_classes=1000, name='resnet152',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4)):
        super().__init__(input_shape=input_shape, num_classes=num_classes, name=name,
                         block_class=ResidualBlockWithBottleNeck, repetitions=(3, 8, 36, 3),
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)





