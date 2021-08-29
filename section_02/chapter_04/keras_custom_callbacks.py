# ===================================================================
# Imported Modules
# ===================================================================

import collections
import tensorflow as tf

# ===================================================================
# Constant Definitions
# ===================================================================

log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\033[94m', '\033[92m'
log_begin_bold = '\033[1m'
log_end_format = '\033[0m'


# ===================================================================
# Class Definitions
# ===================================================================


class SimpleLogCallback(tf.keras.callbacks.Callback):
    """Keras callback for simple, denser console logs."""

    def __init__(self, metrics_dict, num_epochs, log_frequency,
                 metric_string_template='{0}[[name]]{1} = {2}{[[value]]:.4f}{1}'):
        """
        Initialize the callback.
        :param metrics_dict:                  Dictionary containing mappings for metrics names/keys
                                              E.g. {"accuracy": "acc", "val. accuracy": "val_acc"}.
        :param num_epochs:                    Number of training epochs.
        :param log_frequency:                 Log frequency (in epochs).
        :param metric_string_template: (opt.) String template to print each metric.
        """

        super().__init__()

        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency

        # Build format string to later print the metrics, e.g. "Epoch 2/10: loss = 1.8932, val_loss = 3.2312"
        log_string_template = 'Epoch {3:2}/{4}: '
        separator = '; '

        i = 5
        for metric_name in metrics_dict:
            temp = metric_string_template.replace('[[name]]', metric_name)\
                                         .replace('[[value]]', str(i))
            log_string_template += temp + separator
            i += 1

        # Remove '; ' after the last element
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        print('Training: {}start{}'.format(log_begin_red, log_end_format))

    def on_train_end(self, logs=None):
        print('Training: {}end{}'.format(log_begin_green, log_end_format))

    def on_epoch_end(self, epoch, logs={}):
        if ((epoch - 1) % self.log_frequency == 0) or (epoch == self.num_epochs):
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            print(self.log_string_template.format(log_begin_bold, log_end_format, log_begin_blue,
                                                  epoch, self.num_epochs, *values))