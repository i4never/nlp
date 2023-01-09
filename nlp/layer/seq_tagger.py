from typing import List, Tuple, Optional, Callable

import tensorflow as tf

from nlp.layer.utils import get_shape_list
from nlp.layer.activation import get_activation


def crf_outputs(batch_x: tf.Tensor,
                crf_label_n: int,
                activation: Optional[Callable] = None,
                trans: Optional = None,
                get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)):
    """

    :param batch_x: [batch_size, seq_len, hidden_size]
    :param crf_label_n:
    :param activation:
    :param trans: [crf_label_n, crf_label_n]
    :param get_new_initializer_foo:
    :return:
    """
    batch_size, seq_len, hidden_size = get_shape_list(batch_x, expected_rank=3)
    output = tf.layers.dense(
        tf.reshape(batch_x, shape=[batch_size * seq_len, hidden_size]),
        activation=activation,
        units=crf_label_n,
        kernel_initializer=get_new_initializer_foo()
    )
    output = tf.reshape(output, shape=[batch_size, seq_len, crf_label_n])
    if trans is not None:
        crf_trans = tf.get_variable("crf_trans",
                                    initializer=tf.constant(value=trans),
                                    dtype=tf.float32)
    else:
        crf_trans = tf.get_variable("crf_trans",
                                    shape=[crf_label_n, crf_label_n],
                                    initializer=get_new_initializer_foo(),
                                    dtype=tf.float32)
    return output, crf_trans


def global_pointer(batch_encoder_out: tf.Tensor,
                   entity_n: int,
                   size_per_head: int,
                   attention_mask: Optional[tf.Tensor] = None,
                   position_bias: Optional[tf.Tensor] = None,
                   large_negative_number: float = -10000.,
                   get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)):
    batch_size, seq_len, hidden_size = get_shape_list(batch_encoder_out, expected_rank=3)
    # b, s, 2*n*sph
    proj = tf.reshape(
        tf.layers.dense(tf.reshape(batch_encoder_out, shape=[batch_size * seq_len, hidden_size]),
                        units=entity_n * size_per_head * 2,
                        kernel_initializer=get_new_initializer_foo()),
        [batch_size, seq_len, -1]
    )

    # n, b, s, 2*sph
    proj = tf.split(proj, entity_n, axis=-1)
    # b, s, n, 2*sph
    proj = tf.stack(proj, axis=-2)
    # b, s, n, sph
    query_proj, key_proj = proj[..., :size_per_head], proj[..., size_per_head:]

    def transpose_to_bnlh(tensor: tf.Tensor, bs: int, sl: int):
        return tf.transpose(tf.reshape(tensor, shape=[bs, sl, entity_n, size_per_head]), (0, 2, 1, 3))

    # b*n, s, sph
    query_proj = tf.reshape(
        transpose_to_bnlh(query_proj, batch_size, seq_len),
        shape=[batch_size * entity_n, seq_len, size_per_head]
    )
    # b*n, s, sph
    key_proj = tf.reshape(
        transpose_to_bnlh(key_proj, batch_size, seq_len),
        shape=[batch_size * entity_n, seq_len, size_per_head]
    )

    # b*n, s, s
    att_weights = tf.matmul(query_proj, key_proj, transpose_b=True)

    att_weights = tf.reshape(att_weights, shape=[batch_size, entity_n, seq_len, seq_len])
    if attention_mask is not None:
        tf.assert_rank(attention_mask, rank=3,
                       message=f"attention mask shape should be [batch_size, q_len, kv_len]")
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = (1.0 - attention_mask) * large_negative_number
        att_weights += attention_mask

    if position_bias is not None:
        tf.assert_rank(position_bias, rank=3, message=f"position bias shape should be [batch_size, q_len, kv_len]")
        position_bias = tf.expand_dims(position_bias, axis=1)
        att_weights += position_bias
    else:
        # pointer对位置不敏感，需要加上relative position embedding，否则效果一塌糊涂
        from mesoorflow.layer.embedding import sinusoidal_embeddings
        position_bias = sinusoidal_embeddings(tf.range(0, seq_len, dtype=tf.float32),
                                              dim=size_per_head)    # 1, seq_len, size_per_head
        att_weights += position_bias

    # 左下三角mask
    seq_idx = tf.range(seq_len)
    tri_mask = seq_idx[None, :] >= seq_idx[:, None]
    tri_mask = tf.cast(tri_mask, dtype=tf.float32)
    att_weights += (1 - tri_mask) * large_negative_number

    att_weights = att_weights * size_per_head ** -0.5

    # bnll
    att_weights = tf.Print(att_weights, ['gp score shape', tf.shape(att_weights)], summarize=10, first_n=2)
    # # blln
    # att_weights = tf.transpose(att_weights, (0, 2, 3, 1))
    return att_weights, tri_mask

# def isolated_span_outputs(entities: List[str],
#                           batch_encoder_output: tf.Tensor,
#                           activation: str = None,
#                           get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)):
#     batch_size, seq_len, hidden_size = get_shape_list(batch_encoder_output, expected_rank=3)
#     entity_layer_dict = dict()
#     with tf.variable_scope("span"):
#         for entity in entities:
#             with tf.variable_scope(entity):
#                 start = tf.layers.dense(
#                     tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, hidden_size]),
#                     activation=activation,
#                     units=1,
#                     kernel_initializer=get_new_initializer_foo(),
#                     name="start"
#                 )
#                 end = tf.layers.dense(
#                     tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, hidden_size]),
#                     activation=activation,
#                     units=1,
#                     kernel_initializer=get_new_initializer_foo(),
#                     name="end"
#                 )
#                 start = tf.reshape(start, shape=[batch_size, seq_len])
#                 end = tf.reshape(end, shape=[batch_size, seq_len])
#                 start = tf.identity(start, f'start')
#                 end = tf.identity(end, f'end')
#                 entity_layer_dict[entity] = {'start': start,
#                                              'end': end}
#     return entity_layer_dict
#
#
# def span_pointer(batch_encoder_output: tf.Tensor, span_name: str, activation: str = None,
#                  get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)
#                  ) -> Tuple[tf.Tensor, tf.Tensor]:
#     batch_size, seq_len, hidden_size = get_shape_list(batch_encoder_output, expected_rank=3)
#     with tf.variable_scope("span"):
#         with tf.variable_scope(span_name):
#             hidden = tf.layers.dense(
#                 tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, hidden_size]),
#                 activation=activation,
#                 units=hidden_size,
#                 kernel_initializer=get_new_initializer_foo(),
#                 name="fc"
#             )
#             start = tf.layers.dense(
#                 hidden,
#                 activation=None,
#                 units=1,
#                 kernel_initializer=get_new_initializer_foo(),
#                 name="start"
#             )
#             end = tf.layers.dense(
#                 hidden,
#                 activation=None,
#                 units=1,
#                 kernel_initializer=get_new_initializer_foo(),
#                 name="end"
#             )
#             start = tf.reshape(start, shape=[batch_size, seq_len, 1])
#             end = tf.reshape(end, shape=[batch_size, seq_len, 1])
#             output = tf.concat([start, end], axis=-1)
#     return output
