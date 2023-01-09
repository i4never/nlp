from typing import Optional

import tensorflow as tf

from mesoorflow.layer import dropout
from mesoorflow.layer.embedding import word_embedding
from mesoorflow.layer.norm import layer_norm
from mesoorflow.layer.utils import get_shape_list


class GridModel(object):
    def __init__(self, vocab_size: int, embedding_dim: int, base_channel: int, n_anchor: Optional[int] = None,
                 dropout_prob: float = 0.1, norm_type: str = 'layer', is_training: bool = False):
        """
        出于中文vocab太大，one-hot后channel纬度太高的问题，使用一个embedding层让模型学习，embedding作为channel纬度
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.is_training = is_training
        self.reg_scale = 0.1
        self.base_channel = base_channel
        self.n_anchor = n_anchor
        self.dropout_prob = dropout_prob
        self.norm_type = norm_type
        self.reuse = tf.AUTO_REUSE

        self.embedding_table = None

    @staticmethod
    def get_activation(x, name):
        return tf.nn.relu(x, name=name)

    @staticmethod
    def get_new_initializer(stddev: float = 0.02):
        return tf.initializers.he_uniform()

    def dropout(self, x):
        # layer = tf.keras.layers.SpatialDropout2D(rate=self.dropout_prob, data_format='channels_last')
        # return layer(x)
        return dropout(x, dropout_prob=self.dropout_prob)

    def norm(self, x, name):
        if self.norm_type == 'layer':
            return layer_norm(x, name)
        elif self.norm_type == 'batch':
            return tf.layers.batch_normalization(x, training=self.is_training, name=name)
        else:
            raise ValueError(f"{self.norm_type} ?")

    def encode(self, batch_embedded):
        """
        Encoder部分
        :param batch_embedded: [N, H, W, input_C]
        :return:
        """

        outputs = list()

        block_input = batch_embedded
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("encoder_block_0", reuse=self.reuse):
                for i in range(3):
                    x = tf.layers.conv2d(block_input,
                                         filters=self.base_channel, kernel_size=3, strides=2 if i == 0 else 1,
                                         activation=None, padding='same', kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
            x = self.dropout(x)
            outputs.append(x)

        block_input = x
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("encoder_block_1", reuse=self.reuse):
                for i in range(3):
                    x = tf.layers.conv2d(block_input,
                                         filters=2 * self.base_channel, kernel_size=3, strides=2 if i == 0 else 1,
                                         padding='same', kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
            x = self.dropout(x)
            outputs.append(x)

        block_input = x
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("encoder_block_2", reuse=self.reuse):
                for i in range(3):
                    x = tf.layers.conv2d(block_input,
                                         filters=4 * self.base_channel, kernel_size=3,
                                         strides=2 if i == 0 else 1, dilation_rate=1 if i == 0 else 2,
                                         padding='same', kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
            x = self.dropout(x)
            outputs.append(x)

        block_input = x
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("encoder_block_3", reuse=self.reuse):
                for i in range(3):
                    x = tf.layers.conv2d(block_input,
                                         filters=8 * self.base_channel, kernel_size=3, strides=1, dilation_rate=4,
                                         padding='same', kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
            x = self.dropout(x)
            outputs.append(x)

        block_input = x
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("encoder_block_4", reuse=self.reuse):
                for i in range(3):
                    x = tf.layers.conv2d(block_input,
                                         filters=8 * self.base_channel, kernel_size=3,
                                         strides=1, dilation_rate=8, padding='same',
                                         kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
            x = self.dropout(x)
            outputs.append(x)

        return outputs  # [N, H, W, C] [N, H, W, 2C] [N, H, W, 4C] [N, H, W, 8C] [N, H, W, 8C]

    def decode(self, batch_encoder_outputs):
        """

        :param batch_encoder_outputs:
        :return:
        """

        with tf.variable_scope("multimodal", reuse=self.reuse):
            block_input = tf.concat([batch_encoder_outputs[2], batch_encoder_outputs[3]], axis=-1)  # [N, H, W, 12C]
            with tf.variable_scope("bbr_decoder_block_0", reuse=self.reuse):
                x = tf.layers.conv2d(block_input,
                                     filters=4 * self.base_channel, kernel_size=1, strides=1,
                                     padding='same', kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_0")
                x = self.get_activation(x, name=f"activation_0")
                x = self.norm(x, name=f"ln_0")

                x = tf.layers.conv2d_transpose(x,
                                               filters=4 * self.base_channel, kernel_size=3, strides=2, padding='same',
                                               kernel_initializer=self.get_new_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                               name=f"conv_transpose_1")
                x = self.norm(x, name=f"ln_1")
                x = self.get_activation(x, name=f"activation_1")

                x = tf.layers.conv2d(x,
                                     filters=4 * self.base_channel, kernel_size=3, strides=1, padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_2")
                x = self.get_activation(x, name=f"activation_2")
                x = self.norm(x, name=f"ln_2")

                x = tf.layers.conv2d(x,
                                     filters=4 * self.base_channel, kernel_size=3, strides=1, padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_3")
                x = self.get_activation(x, name=f"activation_3")
                x = self.norm(x, name=f"ln_3")
                x = self.dropout(x)

        with tf.variable_scope("multimodal", reuse=self.reuse):
            block_input = tf.concat([batch_encoder_outputs[1], x], axis=-1)  # [N, H, W, 6C]
            with tf.variable_scope("bbr_decoder_block_1", reuse=self.reuse):
                x = tf.layers.conv2d(block_input,
                                     filters=2 * self.base_channel,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_0")
                x = self.get_activation(x, name=f"activation_0")
                x = self.norm(x, name=f"ln_0")

                x = tf.layers.conv2d_transpose(x,
                                               filters=2 * self.base_channel,
                                               kernel_size=3,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=self.get_new_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                               name=f"conv_transpose_1")
                x = self.norm(x, name=f"ln_1")
                x = self.get_activation(x, name=f"activation_1")

                x = tf.layers.conv2d(x,
                                     filters=2 * self.base_channel,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_2")
                x = self.get_activation(x, name=f"activation_2")
                x = self.norm(x, name=f"ln_2")

                x = tf.layers.conv2d(x,
                                     filters=2 * self.base_channel,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_3")
                x = self.get_activation(x, name=f"activation_3")
                x = self.norm(x, name=f"ln_3")
                x = self.dropout(x)

        with tf.variable_scope("multimodal", reuse=self.reuse):
            block_input = tf.concat([batch_encoder_outputs[0], x], axis=-1)  # [N, H, W, 2C]
            with tf.variable_scope("bbr_decoder_block_2", reuse=self.reuse):
                x = tf.layers.conv2d(block_input,
                                     filters=self.base_channel, kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=self.get_new_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                     name=f"conv_0")
                x = self.get_activation(x, name=f"activation_0")
                x = self.norm(x, name=f"ln_0")

                x = tf.layers.conv2d_transpose(x,
                                               filters=self.base_channel, kernel_size=3, strides=2, padding='same',
                                               kernel_initializer=self.get_new_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                               name=f"conv_transpose_1")
                x = self.norm(x, name=f"ln_1")
                x = self.get_activation(x, name=f"activation_1")
                x = self.dropout(x)

        with tf.variable_scope("multimodal", reuse=self.reuse):
            block_input = x
            with tf.variable_scope("pointer_decoder", reuse=self.reuse):
                for i in range(2):
                    x = tf.layers.conv2d(block_input,
                                         filters=self.base_channel, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=self.get_new_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                         name=f"conv_{i}")
                    x = self.get_activation(x, name=f"activation_{i}")
                    x = self.norm(x, name=f"ln_{i}")
                    block_input = x
        return x

    def mlm_output(self, x):
        with tf.variable_scope("mlm", reuse=self.reuse):
            x = tf.layers.conv2d(x,
                                 filters=self.embedding_dim, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=self.get_new_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                 name=f"conv_2")
            x = tf.nn.l2_normalize(x, axis=-1)

            x = tf.matmul(x, self.embedding_table, transpose_b=True)
            output_bias = tf.get_variable("bias", shape=[self.vocab_size], initializer=tf.zeros_initializer())
            x = tf.nn.bias_add(x, output_bias)
            x = tf.identity(x, 'mlm_output')

            print('=' * 10, 'MLM Output', '=' * 10, x)

        return x

    def box_output(self, x):
        """
        标记是否开始/是否结束
        :param x:
        :return:
        """
        with tf.variable_scope("pointer_output_project", reuse=self.reuse):
            x = tf.layers.conv2d(x,
                                 filters=2 * self.n_anchor, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=self.get_new_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                 name=f"conv_2")
            x = tf.identity(x, 'pointer_output')
            print('=' * 10, x)
        return x

    def seg_output(self, x):
        with tf.variable_scope("seg_output_project", reuse=self.reuse):
            x = tf.layers.conv2d(x,
                                 filters=self.n_anchor, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=self.get_new_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale),
                                 name=f"conv_2")
            x = tf.identity(x, 'seg_output')
            print('=' * 10, x)
        return x

    def mlm_loss(self, y_true, y_pred, mask=None):
        batch_size, _, _ = get_shape_list(y_true, expected_rank=3)
        with tf.variable_scope("mlm_loss", reuse=self.reuse):
            loss = tf.losses.sparse_softmax_cross_entropy(
                y_true,
                y_pred,
                weights=mask
            )
            loss = loss / tf.cast(batch_size, tf.float32)
        return loss

    def multilabel_categorical_crossentropy(self, y_true, y_pred, mask=None, thd=0.):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
             本文。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = tf.zeros_like(y_pred[..., :1])
        y_pred_neg = tf.concat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = tf.concat([y_pred_pos, zeros], axis=-1)
        neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
        pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss

    def pointer_multilabel_softmax_loss(self, y_true, y_pred, mask=None):
        """

        :param y_true: [batch_size, H, W, 2 * n_anchor]
        :param y_pred: [batch_size, H, W, 2 * n_anchor]
        :param mask: [batch_size, H, W]
        :return:
        """
        batch_size, H, W, _ = get_shape_list(y_pred, expected_rank=4)
        y_true = y_true.reshape(y_true, (batch_size * H * W, -1))
        y_pred = y_true.reshape(y_pred, (batch_size * H * W, -1))

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = tf.zeros_like(y_pred[..., :1])
        y_pred_neg = tf.concat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = tf.concat([y_pred_pos, zeros], axis=-1)
        neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
        pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss

    def pointer_loss(self, label, predicted, mask=None):
        label = tf.cast(label, tf.float32)
        batch_size, H, W, _ = get_shape_list(predicted, expected_rank=4)
        label = tf.reshape(label, (batch_size * H * W, -1))
        predicted = tf.reshape(predicted, (batch_size * H * W, -1))

        if mask is not None:
            mask = tf.reshape(mask, (batch_size * H * W, 1))
            mask = tf.cast(mask, tf.float32)
        else:
            mask = 1.0

        # 无效：
        # loss = tf.losses.sigmoid_cross_entropy(label, predicted, weights=mask,
        #                                        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        # return loss / (tf.reduce_sum(mask) + 1e-12)

        # weights = label * 999. + 1.
        # weights = 1.0
        loss = tf.losses.sigmoid_cross_entropy(label, predicted, reduction=tf.losses.Reduction.NONE)
        loss = loss * mask
        loss = tf.reshape(loss, shape=[batch_size, H, W, -1])
        loss = tf.reduce_sum(loss, axis=[1, 2])
        return tf.reduce_mean(loss)

    # def anchor_loss(self, label, predicted):
    #     batch_size, H, W, _ = get_shape_list(label, expected_rank=4)
    #     return tf.losses.sigmoid_cross_entropy(tf.reshape(label, (batch_size, -1)),
    #                                            tf.reshape(predicted, (batch_size, -1)))

    # def boundary_loss(self, label, predicted):
    #     """
    #     https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    #     背景不计算
    #     :param label:   batch_size, H, W, 4 * n_anchor
    #     :param predicted:   batch_size, H, W, 4 * n_anchor
    #     :return:
    #     """
    #     mask = tf.cast(tf.not_equal(label[:, :, :, 2], 0), tf.float32)
    #     loss = tf.losses.huber_loss(label, predicted * tf.expand_dims(mask, -1)) / (tf.reduce_sum(mask) + 1e-6)
    #     return 10 * tf.cond(tf.equal(tf.reduce_sum(mask), 0), lambda: 0., lambda: loss)  # 10倍，K.He Faster-RCNN

    def embed(self, batch_image):
        batch_size, H, W = get_shape_list(batch_image, expected_rank=3)

        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("embedding", reuse=self.reuse):
                batch_image = tf.reshape(batch_image, (batch_size, -1))
                embedding_output, self.embedding_table = word_embedding(batch_image, self.vocab_size,
                                                                        self.embedding_dim,
                                                                        get_new_initializer=self.get_new_initializer,
                                                                        name='channel_token_embeddings')
                # output = tf.one_hot(batch_image, depth=self.vocab_size)
                output = tf.reshape(embedding_output, (batch_size, H, W, self.embedding_dim))
                output = self.norm(output, name='embedding_norm')
        return output

    def one_hot_embed(self, batch_image):
        with tf.variable_scope("multimodal", reuse=self.reuse):
            with tf.variable_scope("one_hot_embedding", reuse=self.reuse):
                output = tf.one_hot(batch_image, depth=self.vocab_size)
        return output
