from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as keras_backend

import tensorflow as tf

class ReportingCallback(Callback):
    def __init__(self, summary_names, tensorboard, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tensorboard = tensorboard

        self.summary_placeholders = {}
        self.summary_tensors = []
        for summary_name in summary_names:
            custom_placeholder, custom_tensor = ReportingCallback._construct_summary(summary_name)
            self.summary_placeholders[summary_name] = custom_placeholder
            self.summary_tensors.append(custom_tensor)

        self._custom_summary_mapped_values = None

    def set_custom_summary_values(self, custom_summaries):
        self._custom_summary_mapped_values = {}
        for name, value in custom_summaries.items():
            self._custom_summary_mapped_values[self._find_placeholder_for_name(name)] = value

    def on_epoch_end(self, epoch, logs):
        summary_strings = keras_backend.get_session().run(self.summary_tensors, feed_dict=self._custom_summary_mapped_values)
        for summary_str in summary_strings:
            self._tensorboard.writer.add_summary(summary_str, epoch)

    @staticmethod
    def _construct_summary(summary_name):
        summary_placeholder = tf.placeholder(tf.float32, [], name = summary_name)
        summary_tensor = tf.summary.scalar(summary_name, tensor=summary_placeholder, family='game performance')
        return summary_placeholder, summary_tensor

    def _find_placeholder_for_name(self, placeholder_name):
        return self.summary_placeholders[placeholder_name]