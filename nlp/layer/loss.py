import tensorflow as tf


def iou(y_pred, y_true):
    """
    Returns a (approx) IOU score
        intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    :param y_pred: (4-D array): (N, H, W, n_class)
    :param y_true:  (4-D array): (N, H, W, n_class)
    :return: float: IOU score
    """
    # batch_size, H, W, n_class = y_pred.get_shape().as_list()

    # pred_flat = tf.reshape(y_pred, [batch_size, H * W])
    # true_flat = tf.reshape(y_true, [batch_size, H * W])

    y_true = tf.cast(y_true, tf.float32)
    intersection = 2 * tf.reduce_sum(y_pred * y_true) + 1e-7
    denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + 1e-7

    return tf.reduce_mean(intersection / denominator)
