"""
各种mask，有效位置为1.，无效位置为0.
"""
import tensorflow as tf

from nlp.layer.utils import get_shape_list, assert_rank


def length_mask(batch_token_id: tf.Tensor, pad_value: int = 0, dtype=tf.float32) -> tf.Tensor:
    """
    长度mask
    :param batch_token_id: [batch_size, seq_len]
    :param pad_value:
    :param dtype:
    :return: [batch_size, seq_len, seq_len]
    """
    assert_rank(batch_token_id, expected_rank=2)
    mask = tf.not_equal(batch_token_id, pad_value)
    mask = tf.cast(mask, dtype=dtype)
    mask = mask[:, :, None] * mask[:, None, :]
    return mask


def ar_lm_mask(batch_token_id: tf.Tensor, dtype=tf.float32) -> tf.Tensor:
    """
    下三角attention mask
    :param batch_token_id: [batch_size, seq_length] 或 [batch_size, seq_length, hidden_size] 保证第二个纬度为seq_length即可
    :param dtype:
    :return: [batch_size, seq_length, seq_length]
    """
    input_shape = get_shape_list(batch_token_id, expected_rank=[2, 3])

    sequence_length_index = tf.range(0, input_shape[1])

    mask = sequence_length_index[None, :] <= sequence_length_index[:, None]
    return tf.cast(mask, dtype=dtype)

# def unilm_attention_mask(input_token_ids: tf.Tensor) -> tf.Tensor:
#     """
#     encoder-base生成模型（bert）
#     Args:
#         input_token_ids: [batch_size, seq_length]，0为prefix，1为生成结果
#
#     Returns:
#         bool tensor of shape [batch_size, seq_len, seq_len]
#     """
#     assert_rank(input_token_ids, expected_rank=2)
#     indexes = tf.cumsum(input_token_ids, axis=1)
#     return indexes[:, None, :] <= indexes[:, :, None]
