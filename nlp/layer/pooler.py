from enum import Enum

import tensorflow as tf

from nlp.layer.utils import assert_rank


class PoolerType(Enum):
    CLS = "CLS"
    MEAN = "MEAN"
    MAX = "MAX"


def first_token_pooler(batch_x: tf.Tensor) -> tf.Tensor:
    assert_rank(batch_x, expected_rank=3)
    return batch_x[:, 0, :]


def sequence_mean_pooler(batch_x: tf.Tensor, batch_token_mask: tf.Tensor) -> tf.Tensor:
    assert_rank(batch_x, expected_rank=3)
    assert_rank(batch_token_mask, expected_rank=2)
    output = batch_x * batch_token_mask[:, :, None]
    output = tf.reduce_sum(output, axis=1)
    output = output * (1 / tf.reduce_sum(batch_token_mask, axis=-1, keepdims=True))
    return output


def sequence_max_pooler(batch_x: tf.Tensor, batch_token_mask: tf.Tensor,
                        large_negative_number: float = tf.float32.min) -> tf.Tensor:
    assert_rank(batch_x, expected_rank=3)
    assert_rank(batch_token_mask, expected_rank=2)
    output = batch_x + (1 - batch_token_mask[:, :, None]) * large_negative_number
    output = tf.reduce_max(output, axis=1)
    return output
