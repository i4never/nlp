from typing import Optional

import tensorflow as tf

from mesoorflow.layer import get_shape_list


def mlm_loss(batch_logit: tf.Tensor, batch_label: tf.Tensor, batch_weight: tf.Tensor):
    batch_size, seq_len, vocab_size = get_shape_list(batch_logit, expected_rank=3)
    batch_logit = tf.reshape(batch_logit, shape=[batch_size * seq_len, vocab_size])
    batch_log_probs = tf.nn.log_softmax(batch_logit, axis=-1)
    # batch_log_probs = tf.Print(batch_log_probs, ["batch_log_probs", batch_log_probs], first_n=10)

    batch_label = tf.reshape(batch_label, shape=[-1])
    batch_label_weight = tf.reshape(batch_weight, shape=[-1])
    batch_one_hot_label = tf.one_hot(batch_label, depth=vocab_size, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(batch_log_probs * batch_one_hot_label, axis=-1)
    # per_example_loss = tf.Print(per_example_loss, ["per_example_loss", per_example_loss], first_n=10)
    numerator = tf.reduce_sum(batch_label_weight * per_example_loss)
    # numerator = tf.Print(numerator, ["numerator", numerator], first_n=10)
    denominator = tf.reduce_sum(batch_label_weight) + 1e-5
    # denominator = tf.Print(denominator, ["denominator", denominator], first_n=10)
    loss = numerator / denominator
    # loss = tf.Print(loss, ["loss", loss], first_n=10)
    return loss


def multilabel_categorical_crossentropy(label: tf.Tensor,
                                        logit: tf.Tensor,
                                        weight: Optional[tf.Tensor] = None):
    """多标签分类的交叉熵
    https://github.com/bojone/bert4keras/blob/0cd0a4a0466f244ecc89bed0218dfeaa80fc2c35/bert4keras/backend.py
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064 。
    """
    label = tf.cast(label, tf.float32)
    y_pred = (1 - 2 * label) * logit
    y_pred_neg = y_pred - label * 1e12 - (tf.float32.max * (1. - weight) if weight is not None else 0.)
    y_pred_pos = y_pred - (1 - label) * 1e12 - (tf.float32.max * (1. - weight) if weight is not None else 0.)
    zeros = tf.zeros_like(y_pred[..., :1])
    y_pred_neg = tf.concat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = tf.concat([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)

    neg_loss = tf.Print(neg_loss, ['neg_loss', tf.shape(neg_loss), neg_loss], summarize=10, first_n=1)
    pos_loss = tf.Print(pos_loss, ['pos_loss', tf.shape(pos_loss), pos_loss], summarize=10, first_n=1)
    loss = neg_loss + pos_loss
    loss = tf.Print(loss, ['loss', tf.shape(loss), loss], summarize=10, first_n=1)
    return tf.reduce_mean(loss)


def crf_loss(batch_x: tf.Tensor,
             batch_y: tf.Tensor,
             batch_length: tf.Tensor,
             crf_trans: tf.Tensor):
    """

    :param batch_x: [batch_size, seq_len, num_tags]
    :param batch_y: [batch_size, seq_len] (sparse)
    :param batch_length: [batch_size]
    :param crf_trans: [num_tags, num_tags]
    :return:
    """
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(batch_x,
                                                          batch_y,
                                                          batch_length,
                                                          crf_trans)
    loss = tf.reduce_mean(-log_likelihood)
    return loss
