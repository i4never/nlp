from enum import Enum

from typing import Tuple

import tensorflow as tf

from nlp.layer.utils import get_shape_list, assert_rank


class PositionEmbeddingType(Enum):
    learned_fixed = "learned_fixed"
    learned_relative = "learned_relative"


def word_embedding(batch_token_ids: tf.Tensor, vocab_size: int, embedding_dim: int,
                   get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev),
                   name='word_embeddings') -> Tuple:
    """
    返回look up输出与embedding table
    (以便在mlm pretrain中使用embedding table)
    :param batch_token_ids: [batch_size, seq_len]
    :param vocab_size:
    :param embedding_dim:
    :param get_new_initializer:
    :param name:
    :return: ([batch_size, seq_len, embedding_dim], [vocab_size, embedding_dim])
    """
    batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)
    embedding_table = tf.get_variable(name=name, shape=[vocab_size, embedding_dim], initializer=get_new_initializer())
    flat_input_ids = tf.reshape(batch_token_ids, [-1])
    output = tf.gather(embedding_table, flat_input_ids)
    output = tf.reshape(output, [batch_size, seq_len, embedding_dim])
    return output, embedding_table


def learned_position_embedding(batch_token_ids: tf.Tensor, max_position: int, embedding_dim: int,
                               get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev),
                               name='learned_position_embeddings') -> Tuple:
    """
    返回绝对位置的look up position embedding与position embedding table
    :param batch_token_ids: [batch_size, seq_len]
    :param max_position: 最长位置
    :param embedding_dim:
    :param get_new_initializer:
    :param name:
    :return: ([1, seq_len, embedding_dim], [max_position, embedding_dim])
    """
    batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)
    position_embedding_table = tf.get_variable(name=name, shape=[max_position, embedding_dim],
                                               initializer=get_new_initializer())
    position_embeddings = tf.slice(position_embedding_table, [0, 0], [seq_len, -1])
    position_embeddings = tf.reshape(position_embeddings, [1, seq_len, embedding_dim])
    return position_embeddings, position_embedding_table


def learned_relative_position_embedding_base(batch_query_token_ids: tf.Tensor,
                                             batch_key_value_token_ids: tf.Tensor,
                                             max_position: int,
                                             embedding_dim: int,
                                             get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(
                                                 stddev=stddev),
                                             name="relative_position_embeddings"):
    """
    返回相对位置的look up position embedding与position embedding table
    t位置只与 [t-max_position, t+max_position] 位置有embedding交互，其余超限范围不考虑位置关系
    :param batch_query_token_ids: [batch_size, query_seq_len]
    :param batch_key_value_token_ids: [batch_size, key_value_seq_len]
    :param max_position:
    :param embedding_dim:
    :param get_new_initializer:
    :param name:
    :return: ([batch_size, query_seq_len, kv_seq_len, embedding_dim], [2 * max_position + 1, embedding_dim])
    """
    assert_rank(batch_query_token_ids, expected_rank=2)
    assert_rank(batch_key_value_token_ids, expected_rank=2)
    embedding_table = tf.get_variable(name=name, shape=[2 * max_position + 1, embedding_dim],
                                      initializer=get_new_initializer())

    # bs, q_seq_len, kv_seq_len
    position_ids = tf.expand_dims(batch_key_value_token_ids, 2) - tf.expand_dims(batch_query_token_ids, 1)
    position_ids = tf.clip_by_value(position_ids, -max_position, max_position)
    position_ids = position_ids + max_position
    position_embeddings = tf.gather(embedding_table, position_ids)
    return position_embeddings, embedding_table


def learned_relative_position_embedding(query_seq_len: int,
                                        key_value_seq_len: int,
                                        max_position: int,
                                        embedding_dim: int,
                                        get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(
                                            stddev=stddev),
                                        name='learned_relative_position_embeddings'):
    """
    返回相对位置的look up position embedding与position embedding table
    t位置只与 [t-max_position, t+max_position] 位置有embedding交互，其余超限范围不考虑位置关系
    :param query_seq_len:
    :param key_value_seq_len:
    :param max_position:
    :param embedding_dim:
    :param get_new_initializer:
    :param name:
    :return: ([1, query_seq_len, kv_seq_len, embedding_dim], [2 * max_position + 1, embedding_dim])
    """
    query_index = tf.range(0, query_seq_len)
    key_value_index = tf.range(0, key_value_seq_len)
    # 一般的rpe，只与batch中的seq长度有关，这里填充一个batch_size为1
    batch_query_token_ids = tf.expand_dims(query_index, 0)
    batch_key_value_token_ids = tf.expand_dims(key_value_index, 0)
    return learned_relative_position_embedding_base(batch_query_token_ids, batch_key_value_token_ids, max_position,
                                                    embedding_dim, get_new_initializer, name)


def sinusoidal_embeddings(pos, dim, base=10000):
    """计算pos位置的dim维sinusoidal编码
    """
    _, seq_len = get_shape_list(pos, expected_rank=2)
    assert dim % 2 == 0
    indices = tf.range(0, dim // 2, dtype=tf.float32)
    indices = tf.pow(tf.cast(base, tf.float32), -2 * indices / dim)
    embeddings = tf.einsum('...,d->...d', pos, indices)
    embeddings = tf.stack([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
    embeddings = tf.keras.backend.flatten(embeddings, -2)
    return tf.reshape(embeddings, [1, seq_len, -1])
