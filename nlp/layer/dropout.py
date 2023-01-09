import tensorflow as tf


def dropout(input_tensor: tf.Tensor, dropout_prob: float):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, keep_prob=1.0 - dropout_prob)
    return output
