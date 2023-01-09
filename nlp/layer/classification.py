from typing import Optional, Callable

import tensorflow as tf

from nlp.layer.utils import assert_rank, get_shape_list


def classification_outputs(batch_x: tf.Tensor,
                           units: int,
                           activation: Optional[Callable] = None,
                           get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)):
    """
    （序列）分类
    :param batch_x: [batch_size, hidden_size]
    :param units: 类别数量
    :param activation:
    :param get_new_initializer_foo:
    :return: [batch_size, units]
    """
    assert_rank(batch_x, expected_rank=2)
    output = tf.layers.dense(
        batch_x,
        activation=activation,
        units=units,
        kernel_initializer=get_new_initializer_foo()
    )
    assert_rank(output, expected_rank=2)
    return output


def mlm(batch_encoder_output: tf.Tensor, word_embedding_table: tf.Tensor):
    batch_size, seq_len, _ = get_shape_list(batch_encoder_output, expected_rank=3)
    vocab_size, embedding_dim = get_shape_list(word_embedding_table, expected_rank=2)
    input_tensor = tf.layers.dense(
        tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, embedding_dim]),
        units=embedding_dim,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    input_tensor = tf.nn.l2_normalize(input_tensor, axis=-1)

    output_bias = tf.get_variable("output_bias",
                                  shape=[vocab_size],
                                  initializer=tf.zeros_initializer()
                                  )

    batch_logit = tf.matmul(input_tensor, word_embedding_table, transpose_b=True)
    batch_logit = tf.nn.bias_add(batch_logit, output_bias)
    return tf.reshape(batch_logit, shape=[batch_size, seq_len, vocab_size])
