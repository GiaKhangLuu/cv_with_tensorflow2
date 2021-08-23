#===================================================================
# Imported Modules
#===================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import functools

#===================================================================
# Constant Definitions
#===================================================================

CIFAR100_BUILDER = tfds.builder('cifar100')
CIFAR100_BUILDER.download_and_prepare()

#===================================================================
# Function Definitions
#===================================================================

def _prepare_data_fn(features, input_shape, augment=False, return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimension, and opt. apply some random transformation.
    :param features:              Data
    :param input_shape:           Shape expected by the model (image will be resized accordingly)
    :param augment:               Flag to apply some random augmentations to the image
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """

    # Convert input shape to  tensor
    input_shape = tf.convert_to_tensor(input_shape)

    image = features['image']
    # Convert image to float type & scale pixel values from [0, 255] to [0., 1.]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if augment:
        # Random apply horizontal flip
        image = tf.image.random_flip_left_right(image, seed=seed)

        # Random bright & saturation change
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)

        # Keeping pixel values in check
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        # Random resize and random crop image back to expected size
        random_scale = tf.random.uniform([], minval=1., maxval=1.4, seed=seed)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale, tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale, tf.int32)
        scaled_shape = tf.stack(scaled_height, scaled_width)
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape, seed=seed)
    else:
        # Resize image to expected size
        image = tf.image.resize(image, input_shape[:2])

    if return_batch_as_tuple:
        label = features['label']
        features = (image, label)
    else:
        features['image'] = image
    return features


def get_info():
    """
    Return the Tensorflow-Dataset info for CIFAR-100.
    :return: CIFAR-100 info
    """
    return CIFAR100_BUILDER.info


def get_dataset(phase='train', batch_size=32, num_epochs=None, shuffle=True, input_shape=(32, 32, 3),
                return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param input_shape:           Shape of the processed images
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operation
    :return:                      Iterable dataset
    """

    assert phase == 'train' or phase == 'val'
    is_train = phase == 'train'

    prepare_data_fn = functools.partial(_prepare_data_fn, return_batch_as_tuple=return_batch_as_tuple,
                                        input_shape=input_shape, augment=is_train, seed=seed)

    cifar_dataset = CIFAR100_BUILDER.as_dataset(split='train' if phase == 'train' else 'test')
    cifar_dataset = cifar_dataset.repeat(num_epochs)
    if shuffle:
        cifar_dataset = cifar_dataset.shuffle(1000, seed)
    cifar_dataset = cifar_dataset.map(prepare_data_fn, num_parallel_calls=tf.data.AUTOTUNE)
    cifar_dataset = cifar_dataset.batch(batch_size)
    cifar_dataset = cifar_dataset.prefetch(tf.data.AUTOTUNE)
    return cifar_dataset