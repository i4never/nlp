import math
import tensorflow as tf
from typing import Optional, Tuple

from nlp.layer.utils import get_shape_list, reshape_to_matrix


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev)):
    """
    From google bert
    :param from_tensor:
    :param to_tensor:
    :param attention_mask:
    :param num_attention_heads:
    :param size_per_head:
    :param query_act:
    :param key_act:
    :param value_act:
    :param attention_probs_dropout_prob:
    :param do_return_2d_tensor:
    :param batch_size:
    :param from_seq_length:
    :param to_seq_length:
    :param get_new_initializer:
    :return:
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=get_new_initializer())

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=get_new_initializer())

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=get_new_initializer())

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    from mesoorflow.layer.dropout import dropout
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def multi_head_attention(query: tf.Tensor,
                         key_value: Optional[tf.Tensor] = None,
                         attention_mask: Optional[tf.Tensor] = None,
                         position_bias: Optional[tf.Tensor] = None,
                         cache_key_value_states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
                         hidden_size: int = 768,
                         attention_head_n: int = 12,
                         size_per_head: int = 64,
                         attention_probs_dropout_prob: float = 0.2,
                         query_act: Optional[str] = None,
                         key_act: Optional[str] = None,
                         value_act: Optional[str] = None,
                         large_negative_number: float = tf.float32.min,
                         get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev)):
    """
    支持缓存的MHA，在decoder ar生成时可以减少cross/self-att的计算时间，也可以作为普通的MHA使用
        - 有query，无key_value
            - 无cache_key_value_states 无缓存 self-attention
            - 有cache_key_value_states 有缓存 self-attention 节省key/value的project运算
        - 有query，无key_value
            - 无cache_key_value_states 无缓存 cross-attention
            - 有cache_key_value_states 有缓存 cross-attention 节省key/value的softmax运算
    :param query: [batch_size, q_len, hidden_size]
    :param key_value: [batch_size, mv_len, hidden_size]
    :param attention_mask: [batch_size, q_len, kv_len]
    :param position_bias: [batch_size, q_len, kv_len]
    :param cache_key_value_states: ([batch_size, kv_len - 1, hidden_size], [batch_size, kv_len - 1, hidden_size])
    :param hidden_size:
    :param attention_head_n:
    :param size_per_head:
    :param attention_probs_dropout_prob:
    :param query_act:
    :param key_act:
    :param value_act:
    :param large_negative_number: 计算att score时的softmax mask
    :param get_new_initializer:
    :return:
    """
    is_cross_attention = key_value is not None

    batch_size, q_len, q_hidden_size = get_shape_list(query, expected_rank=3)

    if is_cross_attention:
        batch_size, kv_len, kv_hidden_size = get_shape_list(key_value, expected_rank=3)
        assert q_hidden_size == kv_hidden_size == hidden_size
    else:
        assert q_hidden_size == hidden_size

    assert size_per_head * attention_head_n == hidden_size

    scale = size_per_head ** -0.5
    query_proj = tf.reshape(
        tf.layers.dense(tf.reshape(query, [batch_size * q_len, hidden_size]),
                        units=attention_head_n * size_per_head,
                        activation=query_act,
                        name="query",
                        kernel_initializer=get_new_initializer()),
        [batch_size, q_len, hidden_size]
    )
    query_proj *= scale

    # ======== [获取kv states] START ========
    if is_cross_attention:
        if cache_key_value_states is not None:
            # cross attention，有缓存
            key_proj = cache_key_value_states[0]
            value_proj = cache_key_value_states[1]
        else:
            # cross attention，无缓存
            key_proj = tf.reshape(
                tf.layers.dense(tf.reshape(key_value, [batch_size * kv_len, hidden_size]),
                                units=attention_head_n * size_per_head,
                                activation=key_act,
                                name="key",
                                kernel_initializer=get_new_initializer()),
                [batch_size, kv_len, hidden_size]
            )
            value_proj = tf.reshape(
                tf.layers.dense(tf.reshape(key_value, [batch_size, kv_len, hidden_size]),
                                units=attention_head_n * size_per_head,
                                activation=value_act,
                                name="value",
                                kernel_initializer=get_new_initializer()),
                [batch_size, kv_len, hidden_size]
            )
    else:
        kv_len = q_len
        # self attention，无缓存
        key_proj = tf.reshape(
            tf.layers.dense(tf.reshape(query, [batch_size * q_len, hidden_size]),
                            units=attention_head_n * size_per_head,
                            activation=key_act,
                            name="key",
                            kernel_initializer=get_new_initializer()),
            [batch_size, q_len, hidden_size]
        )
        value_proj = tf.reshape(
            tf.layers.dense(tf.reshape(query, [batch_size, q_len, hidden_size]),
                            units=attention_head_n * size_per_head,
                            activation=value_act,
                            name="value",
                            kernel_initializer=get_new_initializer()),
            [batch_size, q_len, hidden_size]
        )
        if cache_key_value_states is not None:
            # self attention，有缓存（仅在decoder的情况下可能用到）
            key_proj = tf.concat([cache_key_value_states[0], key_proj], axis=1)
            value_proj = tf.concat([cache_key_value_states[1], value_proj], axis=1)
            kv_len = get_shape_list(key_proj, expected_rank=3)[1]

    updated_cache_key_value_states = (key_proj, value_proj)

    # query_proj: [bs, q_len, H]  key_proj: [bs, kv_len, H]  value_prod: [bs, kv_len, H]
    # ======== [获取kv states] END ========

    # ======== [计算attention score] START ========
    def transpose_to_bnlh(tensor: tf.Tensor, bs: int, sl: int):
        return tf.transpose(tf.reshape(tensor, shape=[bs, sl, attention_head_n, size_per_head]), (0, 2, 1, 3))

    # b*n, q, h
    query_proj = tf.reshape(
        transpose_to_bnlh(query_proj, batch_size, q_len),
        shape=[batch_size * attention_head_n, q_len, size_per_head]
    )
    # b*n, kv, h
    key_proj = tf.reshape(
        transpose_to_bnlh(key_proj, batch_size, kv_len),
        shape=[batch_size * attention_head_n, kv_len, size_per_head]
    )
    value_proj = tf.reshape(
        transpose_to_bnlh(value_proj, batch_size, kv_len),
        shape=[batch_size * attention_head_n, kv_len, size_per_head]
    )

    # b*n, q, kv
    att_weights = tf.matmul(query_proj, key_proj, transpose_b=True)

    # attention_mask & position_bias
    att_weights = tf.reshape(att_weights, shape=[batch_size, attention_head_n, q_len, kv_len])

    # b, q_len, kv_len
    if position_bias is not None:
        tf.assert_rank(position_bias, rank=3, message=f"position bias shape should be [batch_size, q_len, kv_len]")
        position_bias = tf.expand_dims(position_bias, axis=1)
        att_weights += position_bias

    # b, q_len, kv_len
    if attention_mask is not None:
        tf.assert_rank(attention_mask, rank=3, message=f"attention mask shape should be [batch_size, q_len, kv_len]")
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = (1.0 - attention_mask) * large_negative_number
        att_weights += attention_mask

    att_weights = tf.reshape(att_weights, shape=[batch_size * attention_head_n, q_len, kv_len])
    att_probs = tf.nn.softmax(att_weights, -1)

    from mesoorflow.layer.dropout import dropout
    att_probs = dropout(att_probs, attention_probs_dropout_prob)
    # ======== [计算attention prob] END ========

    # b*n, q_len, h
    # return tf.reshape(att_probs, [batch_size, attention_head_n, q_len, kv_len]), att_probs, att_probs
    att_output = tf.matmul(att_probs, value_proj)
    att_output = tf.transpose(
        tf.reshape(att_output, shape=(batch_size, attention_head_n, q_len, size_per_head)),
        (0, 2, 1, 3)
    )
    att_output = tf.reshape(att_output, shape=[batch_size, q_len, attention_head_n * size_per_head])
    return (
        att_output,
        tf.reshape(att_weights, shape=(batch_size, attention_head_n, q_len, kv_len)),
        updated_cache_key_value_states
    )


def gated_attention(query: tf.Tensor,
                    key_value: Optional[tf.Tensor] = None,
                    attention_mask: Optional[tf.Tensor] = None,
                    position_bias: Optional[tf.Tensor] = None,
                    hidden_size: int = 768,
                    attention_head_n: int = 12,
                    size_per_head: int = 64,
                    attention_probs_dropout_prob: float = 0.2,
                    query_act: Optional[str] = None,
                    key_act: Optional[str] = None,
                    value_act: Optional[str] = None,
                    get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev)):
    is_cross_attention = key_value is not None
    batch_size, q_len, q_hidden_size = get_shape_list(query, expected_rank=3)

    if is_cross_attention:
        _, kv_len, kv_hidden_size = get_shape_list(key_value, expected_rank=3)
        assert q_hidden_size == kv_hidden_size == hidden_size
    else:
        assert q_hidden_size == hidden_size

    assert size_per_head * attention_head_n == hidden_size


def linear_attention_layer(inputs,
                           attention_mask,
                           unit_size,
                           key_size,
                           num_attention_heads,
                           hidden_act_fn,
                           get_new_initializer=lambda stddev=0.02: tf.truncated_normal_initializer(stddev=stddev)):
    batch_size, seq_length, _ = get_shape_list(inputs, expected_rank=3)

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    hidden_size = inputs.shape.as_list()[-1]

    short_cut, inputs = inputs, inputs

    attention_mask = tf.reshape(attention_mask, [batch_size, 1, seq_length, 1])

    u_layer, v_layer, query_layer, key_layer = tf.split(
        tf.layers.dense(inputs, 2 * num_attention_heads * unit_size + 2 * num_attention_heads * key_size,
                        kernel_initializer=get_new_initializer()),
        [num_attention_heads * unit_size,
         num_attention_heads * unit_size,
         num_attention_heads * key_size,
         num_attention_heads * key_size], axis=-1)

    v_layer = hidden_act_fn(v_layer)
    v_layer = transpose_for_scores(v_layer, batch_size,
                                   num_attention_heads, seq_length,
                                   unit_size) * attention_mask

    u_layer = hidden_act_fn(u_layer)
    u_layer = transpose_for_scores(u_layer, batch_size,
                                   num_attention_heads, seq_length,
                                   unit_size) * attention_mask

    query_layer = tf.nn.relu(query_layer)
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, seq_length,
                                       key_size) * attention_mask

    key_layer = tf.nn.relu(key_layer)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     seq_length, key_size) * attention_mask

    v = tf.matmul(query_layer, tf.matmul(tf.transpose(key_layer, [0, 1, 3, 2]), v_layer)) / \
        (tf.matmul(query_layer, tf.expand_dims(tf.reduce_sum(key_layer, 2), -1)) + 1e-12)

    v = tf.contrib.layers.layer_norm(
        inputs=v, begin_norm_axis=-1, begin_params_axis=-1, scope=None)

    context_layer = u_layer * v

    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    context_layer = tf.reshape(
        context_layer,
        [batch_size, seq_length, num_attention_heads * unit_size])

    return tf.layers.dense(context_layer, hidden_size, kernel_initializer=get_new_initializer())
