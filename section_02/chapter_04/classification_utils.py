# ===================================================================
# Imported Modules
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ===================================================================
# Function Definitions
# ===================================================================


def load_image(image_path, size):
    """
    Load an image as a Numpy array.
    :param image_path: Path of the image.
    :param size:       Target size.
    :return:           Image array, normalized between 0 and 1.
    """

    images = img_to_array(load_img(image_path, size)) / 255.
    return images


def process_predictions(class_probabilities, class_readable_labels, k=5):
    """
    Process a batch of predictions from our estimator.
    :param class_probabilities:   Prediction results returned by the Keras classifier for a batch of data.
    :param class_readable_labels: List of readable labels, for display.
    :param k:                     Number of predictions to consider.
    :return:                      Readable labels and probabilities for the predicted classes.
    """

    topk_labels, topk_probabilities = [], []
    for i in range(len(class_probabilities)):
        # Getting indexes of k top predictions, dont need to order
        topk_predictions = sorted(np.argpartition(class_probabilities[i], -k)[-k:])

        # Getting the corresponding labels & probabilities
        topk_labels.append([class_readable_labels[prediction] for prediction in topk_predictions])
        topk_probabilities.append(class_probabilities[i][topk_predictions])
    return topk_labels, topk_probabilities


def display_predictions(images, topk_labels, topk_probabilities):
    """
    Plot a batch of predictions
    :param images:             Batch of input images.
    :param topk_labels:        String labels of predicted class.
    :param topk_probabilities: Probabilities for each class.
    :return:
    """

    num_imgs = len(images)
    num_imgs_sqrt = np.sqrt(num_imgs)
    num_rows = num_cols = int(np.ceil(num_imgs_sqrt))

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)  # `constrained_layout` automatically adjust subplots
    grid_spec = gridspec.GridSpec(num_rows, num_cols)

    for i in range(num_imgs):
        img, pred_labels, pred_probabilities = images[i], topk_labels[i], topk_probabilities[i]

        # Gridspec inside gridspec, create layout
        grid_spec_i = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid_spec[i])
        ax_img = fig.add_subplot(grid_spec_i[:-1])
        ax_pred = fig.add_subplot(grid_spec_i[-1])

        # Drawing the input image
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.autoscale(tight=True)

        # Plotting a bar chart for prediction
        y_pos = np.arange(len(pred_probabilities))
        ax_pred.barh(y_pos, pred_probabilities)
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['left'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['bottom'].set_visible(False)
        ax_pred.set_yticks(y_pos)
        ax_pred.set_yticklabels(pred_labels)
        ax_pred.invert_yaxis()  # labels read top-to-bottom

    plt.tight_layout()  # `tight_layout()` similar to `constrained_layout = True`
    plt.show()