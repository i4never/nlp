import tensorflow as tf
from pydantic import validator
from typing import Optional, Dict, Any, List, Tuple

from nlp.layer.activation import get_activation
from nlp.layer.attention import linear_attention_layer
from nlp.layer.dropout import dropout
from nlp.layer.embedding import word_embedding, PositionEmbeddingType, learned_position_embedding
from nlp.layer.norm import layer_norm
from nlp.layer.pooler import first_token_pooler, sequence_mean_pooler, sequence_max_pooler, PoolerType
from nlp.layer.utils import get_shape_list
from nlp.loss import mlm_loss
from nlp.model.model import Model


@Model.register("bert_2d_linear", exist_ok=True)
class Bert2DLinear(Model):
    vocab_size: int = 21128
    page_vocab_size: int = 2
    width_vocab_size: int = 1000
    height_vocab_size: int = 1000
    embedding_dim: int = 768
    hidden_size: int = 768
    attention_head_n: int = 12
    size_per_head: int = 64
    encoder_layer_n: int = 12
    dropout_rate: float = 0.2
    hidden_act: str = 'silu'
    pad_token_id: int = 0
    normal_initializer_stddev: float = 0.02
    pooler: Optional[PoolerType] = None

    def __init__(self, **kwargs):
        super(Bert2DLinear, self).__init__(**kwargs)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """

        :param inputs: (batch_token_ids, batch_x_left_ids, batch_x_right_ids, batch_y_top_ids, batch_y_bottom_ids, batch_page_ids)
        :return: (最后一层pooled输出，所有block输出，embedding lookup表)
        """
        batch_token_ids = inputs['batch_token_ids']

        with tf.variable_scope("bert_2d", reuse=tf.AUTO_REUSE):
            batch_length_mask = tf.cast(
                tf.not_equal(batch_token_ids, self.pad_token_id),
                tf.float32
            )  # batch_size, seq_len
            with tf.variable_scope("embeddings"):
                batch_embedded, embedding_table = self.embed(batch_token_ids,
                                                             inputs['batch_x_left_ids'],
                                                             inputs['batch_x_right_ids'],
                                                             inputs['batch_y_top_ids'],
                                                             inputs['batch_y_bottom_ids'],
                                                             inputs['batch_page_ids'])

            with tf.variable_scope("encoder"):
                all_batch_encoded = self.encode(batch_embedded, batch_length_mask)

            with tf.variable_scope("pooler"):
                batch_encoder_out = self.pool(all_batch_encoded[-1], batch_length_mask)
        outputs = {"batch_encoder_out": batch_encoder_out, "embedding_table": embedding_table}
        for idx, layer_out in enumerate(all_batch_encoded):
            outputs[f"transformer/block_{idx}"] = layer_out
        return outputs

    def get_new_initializer(self):
        return tf.truncated_normal_initializer(stddev=self.normal_initializer_stddev)

    def embed(self, batch_token_ids: tf.Tensor, batch_x_left_ids: tf.Tensor, batch_x_right_ids: tf.Tensor,
              batch_y_top_ids: tf.Tensor, batch_y_bottom_ids: tf.Tensor, batch_page_ids: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)

        token_embedding_output, token_embedding_table = word_embedding(batch_token_ids, self.vocab_size,
                                                                       self.embedding_dim,
                                                                       get_new_initializer=self.get_new_initializer,
                                                                       name='word_embeddings')

        x_embedding_table = tf.get_variable(name='x_embeddings', shape=[self.width_vocab_size, self.embedding_dim],
                                            initializer=self.get_new_initializer())
        y_embedding_table = tf.get_variable(name='y_embeddings', shape=[self.height_vocab_size, self.embedding_dim],
                                            initializer=self.get_new_initializer())
        page_embedding_table = tf.get_variable(name='page_embeddings', shape=[self.page_vocab_size, self.embedding_dim],
                                               initializer=self.get_new_initializer())

        x_left_embedding_output = tf.gather(x_embedding_table, tf.reshape(batch_x_left_ids, [-1]))
        x_left_embedding_output = tf.reshape(x_left_embedding_output, [batch_size, seq_len, self.embedding_dim])

        x_right_embedding_output = tf.gather(x_embedding_table, tf.reshape(batch_x_right_ids, [-1]))
        x_right_embedding_output = tf.reshape(x_right_embedding_output, [batch_size, seq_len, self.embedding_dim])

        y_top_embedding_output = tf.gather(y_embedding_table, tf.reshape(batch_y_top_ids, [-1]))
        y_top_embedding_output = tf.reshape(y_top_embedding_output, [batch_size, seq_len, self.embedding_dim])

        y_bottom_embedding_output = tf.gather(y_embedding_table, tf.reshape(batch_y_bottom_ids, [-1]))
        y_bottom_embedding_output = tf.reshape(y_bottom_embedding_output, [batch_size, seq_len, self.embedding_dim])

        page_embedding_output = tf.gather(page_embedding_table, tf.reshape(batch_page_ids, [-1]))
        page_embedding_output = tf.reshape(page_embedding_output, [batch_size, seq_len, self.embedding_dim])

        output = token_embedding_output + x_left_embedding_output + x_right_embedding_output + \
                 y_top_embedding_output + y_bottom_embedding_output + page_embedding_output

        output = layer_norm(output, 'norm_embedding')
        output = dropout(output, self.dropout_prob)
        return output, token_embedding_table

    def encode(self, batch_embedded: tf.Tensor, batch_length_mask: tf.Tensor,
               attention_mask: Optional[tf.Tensor] = None) -> List[tf.Tensor]:
        batch_size, seq_len, _ = get_shape_list(batch_embedded, expected_rank=3)

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32) * tf.cast(batch_length_mask, tf.float32)
        else:
            attention_mask = batch_length_mask

        all_encoder_outputs = list()

        prev_output = batch_embedded
        for idx in range(self.encoder_layer_n):
            with tf.variable_scope(f"layer_{idx}"):
                with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                    with tf.variable_scope("self"):
                        att_output = linear_attention_layer(inputs=prev_output,
                                                            attention_mask=attention_mask,
                                                            unit_size=self.size_per_head,
                                                            key_size=self.size_per_head,
                                                            num_attention_heads=self.attention_head_n,
                                                            hidden_act_fn=get_activation('silu'),
                                                            get_new_initializer=self.get_new_initializer)

                        att_output = dropout(att_output, self.dropout_prob)
                        layer_output = layer_norm(prev_output + att_output)
                        layer_output = tf.reshape(layer_output, shape=[batch_size, seq_len, self.hidden_size])
                        prev_output = layer_output
                        all_encoder_outputs.append(layer_output)
        return all_encoder_outputs

    def pool(self, batch_encoded: tf.Tensor, batch_token_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self.pooler is None:
            return batch_encoded
        elif self.pooler == PoolerType.CLS:
            return first_token_pooler(batch_encoded)
        elif self.pooler == PoolerType.MEAN:
            return sequence_mean_pooler(batch_encoded, batch_token_mask)
        elif self.pooler == PoolerType.MAX:
            return sequence_max_pooler(batch_encoded, batch_token_mask)
        else:
            raise ValueError(f"不支持pooler: {self.pooler}")

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        raise NotImplementedError(
            f"{self.__class__.__name__}只包含encoder部分，没有具体任务定义，需要使用MLM或者其他任务加载bert模型")

    def _to_params(self) -> Dict[str, Any]:
        params = self.dict()
        params['type'] = 'bert'
        return params


@Model.register("bert_2d_linear_mlm", exist_ok=True)
class Bert2DLinearMLM(Model):
    l2_normalize: bool = True
    encoder: Model

    class Config:
        arbitrary_types_allowed = True

    @validator("encoder", pre=True)
    def load_encoder(cls, encoder_params):
        return Model.from_params(**encoder_params)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        if isinstance(self.encoder, Bert2DLinear):
            encoder_output = self.encoder.forward(inputs)
            batch_encoder_out = encoder_output['batch_encoder_out']
            embedding_table = encoder_output['embedding_table']
        else:
            raise ValueError(f"{self.__class__} 没有适配encoder: {self.encoder.__class__}")

        batch_size, seq_len, _ = get_shape_list(batch_encoder_out, expected_rank=3)

        with tf.variable_scope("mlm", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    tf.reshape(batch_encoder_out, shape=[batch_size * seq_len, self.encoder.hidden_size]),
                    units=self.encoder.hidden_size,
                    activation=None,
                    kernel_initializer=self.encoder.get_new_initializer())
                if self.l2_normalize:
                    input_tensor = tf.nn.l2_normalize(input_tensor, axis=-1)

            output_bias = tf.get_variable("output_bias",
                                          shape=[self.encoder.vocab_size],
                                          initializer=tf.zeros_initializer())

            batch_logits = tf.matmul(input_tensor, embedding_table, transpose_b=True)
            batch_logits = tf.nn.bias_add(batch_logits, output_bias)
            batch_logits = tf.reshape(batch_logits, [batch_size, seq_len, self.encoder.vocab_size])

            batch_token_ids = tf.argmax(batch_logits, axis=-1)

        return {"batch_logits": batch_logits, "batch_token_ids": batch_token_ids}

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            loss = mlm_loss(outputs['batch_logits'], labels['batch_label_ids'], weights['batch_label_weights'])
        return loss

    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        acc = tf.metrics.accuracy(labels=outputs["batch_token_ids"],
                                  predictions=labels["batch_label_ids"],
                                  weights=weights["batch_label_weights"])
        recall = tf.metrics.recall(labels=outputs["batch_token_ids"],
                                   predictions=labels["batch_label_ids"],
                                   weights=weights["batch_label_weights"])
        return {"acc": acc, "recall": recall}


@Model.register("bert_linear", exist_ok=True)
class BertLinear(Model):
    vocab_size: int = 21128
    type_vocab_size: int = 2
    embedding_dim: int = 768
    hidden_size: int = 768
    attention_head_n: int = 12
    size_per_head: int = 64
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_fixed
    max_position: int = 16384
    encoder_layer_n: int = 12
    dropout_rate: float = 0.2
    hidden_act: str = 'silu'
    pad_token_id: int = 0
    normal_initializer_stddev: float = 0.02
    pooler: Optional[PoolerType] = None

    def __init__(self, **kwargs):
        super(BertLinear, self).__init__(**kwargs)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """

        :param inputs: (batch_token_ids)
        :return: (最后一层pooled输出，所有block输出，embedding lookup表)
        """
        batch_token_ids = inputs['batch_token_ids']

        with tf.variable_scope("bert_linear", reuse=tf.AUTO_REUSE):
            batch_length_mask = tf.cast(
                tf.not_equal(batch_token_ids, self.pad_token_id),
                tf.float32
            )  # batch_size, seq_len
            with tf.variable_scope("embeddings"):
                batch_embedded, embedding_table = self.embed(batch_token_ids,
                                                             inputs.get("batch_type_ids"))

            with tf.variable_scope("encoder"):
                all_batch_encoded = self.encode(batch_embedded, batch_length_mask)

            with tf.variable_scope("pooler"):
                batch_encoder_out = self.pool(all_batch_encoded[-1], batch_length_mask)
        outputs = {"batch_encoder_out": batch_encoder_out, "embedding_table": embedding_table}
        for idx, layer_out in enumerate(all_batch_encoded):
            outputs[f"transformer/block_{idx}"] = layer_out
        return outputs

    def get_new_initializer(self):
        return tf.truncated_normal_initializer(stddev=self.normal_initializer_stddev)

    def embed(self, batch_token_ids: tf.Tensor, batch_type_ids: Optional[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)

        token_embedding_output, token_embedding_table = word_embedding(batch_token_ids, self.vocab_size,
                                                                       self.embedding_dim,
                                                                       get_new_initializer=self.get_new_initializer,
                                                                       name='word_embeddings')
        output = token_embedding_output

        if self.type_vocab_size > 1:
            type_embedding_output, _ = word_embedding(batch_type_ids, self.type_vocab_size, self.embedding_dim,
                                                      get_new_initializer=self.get_new_initializer,
                                                      name='type_embeddings')
            output += type_embedding_output

        with tf.control_dependencies([tf.assert_less_equal(seq_len, self.max_position)]):
            position_embedding_output, _ = learned_position_embedding(batch_token_ids, self.max_position,
                                                                      self.embedding_dim,
                                                                      get_new_initializer=self.get_new_initializer,
                                                                      name='learned_position_embeddings')

        output += position_embedding_output
        output = layer_norm(output, 'norm_embedding')
        output = dropout(output, self.dropout_prob)
        return output, token_embedding_table

    def encode(self, batch_embedded: tf.Tensor, batch_length_mask: tf.Tensor,
               attention_mask: Optional[tf.Tensor] = None) -> List[tf.Tensor]:
        batch_size, seq_len, _ = get_shape_list(batch_embedded, expected_rank=3)

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32) * tf.cast(batch_length_mask, tf.float32)
        else:
            attention_mask = batch_length_mask

        all_encoder_outputs = list()

        prev_output = batch_embedded
        for idx in range(self.encoder_layer_n):
            with tf.variable_scope(f"layer_{idx}"):
                with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                    with tf.variable_scope("self"):
                        att_output = linear_attention_layer(inputs=prev_output,
                                                            attention_mask=attention_mask,
                                                            unit_size=self.size_per_head,
                                                            key_size=self.size_per_head,
                                                            num_attention_heads=self.attention_head_n,
                                                            hidden_act_fn=get_activation('silu'),
                                                            get_new_initializer=self.get_new_initializer)

                        att_output = dropout(att_output, self.dropout_prob)
                        layer_output = layer_norm(prev_output + att_output)
                        layer_output = tf.reshape(layer_output, shape=[batch_size, seq_len, self.hidden_size])
                        prev_output = layer_output
                        all_encoder_outputs.append(layer_output)
        return all_encoder_outputs

    def pool(self, batch_encoded: tf.Tensor, batch_token_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self.pooler is None:
            return batch_encoded
        elif self.pooler == PoolerType.CLS:
            return first_token_pooler(batch_encoded)
        elif self.pooler == PoolerType.MEAN:
            return sequence_mean_pooler(batch_encoded, batch_token_mask)
        elif self.pooler == PoolerType.MAX:
            return sequence_max_pooler(batch_encoded, batch_token_mask)
        else:
            raise ValueError(f"不支持pooler: {self.pooler}")

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        raise NotImplementedError(
            f"{self.__class__.__name__}只包含encoder部分，没有具体任务定义，需要使用MLM或者其他任务加载bert模型")

    def _to_params(self) -> Dict[str, Any]:
        params = self.dict()
        params['type'] = 'bert'
        return params


if __name__ == '__main__':
    import numpy as np
    import tqdm

    bert_linear = Model.from_params(
        **{
            "name": "bert_linear",
            "vocab_size": 2345,
            "type_vocab_size": 2,
            "embedding_dim": 768,
            "hidden_size": 768,
            "attention_head_n": 12,
            "size_per_head": 64,
            "max_position": 16384,
            "encoder_layer_n": 6,
            "dropout_rate": 0.2,
            "hidden_act": "silu"
        })
    print(bert_linear)

    token_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    type_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)

    out = bert_linear.forward({"batch_token_ids": token_phd,
                               "batch_type_ids": type_phd})

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        for _ in tqdm.tqdm(range(10)):
            o = sess.run(out, feed_dict={
                token_phd: np.random.randint(0, 2344, (3, 10240)),
                type_phd: np.random.randint(0, 1, (3, 10240))
            })
            print(o)
            for k, v in o.items():
                print(k, v.shape)

    # bert_2d_linear = Model.from_params(
    #     **{
    #         "name": "bert_2d_linear",
    #         "vocab_size": 2345,
    #         "page_vocab_size": 4,
    #         "width_vocab_size": 1000,
    #         "height_vocab_size": 1000,
    #         "embedding_dim": 768,
    #         "hidden_size": 768,
    #         "attention_head_n": 12,
    #         "size_per_head": 64,
    #         "encoder_layer_n": 6,
    #         "dropout_rate": 0.2,
    #         "hidden_act": "silu"
    #     })
    # print(bert_2d_linear)
    #
    # token_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    # x_left_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    # x_right_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    # y_top_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    # y_bottom_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    # page_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    #
    # out = bert_2d_linear.forward({"batch_token_ids": token_phd,
    #                               "batch_page_ids": page_phd,
    #                               "batch_x_left_ids": x_left_phd,
    #                               "batch_x_right_ids": x_right_phd,
    #                               "batch_y_top_ids": y_top_phd,
    #                               "batch_y_bottom_ids": y_bottom_phd})
    #
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     for _ in range(10):
    #         ids = np.random.randint(0, 2344, (5, 27))
    #         coord_ids = np.random.randint(0, 9, (5, 27))
    #         length = np.random.randint(1, 27, (5,))
    #         o = sess.run(out, feed_dict={
    #             token_phd: np.random.randint(0, 2344, (3, 29)),
    #             x_left_phd: np.random.randint(0, 1000, (3, 29)),
    #             x_right_phd: np.random.randint(0, 1000, (3, 29)),
    #             y_top_phd: np.random.randint(0, 1000, (3, 29)),
    #             y_bottom_phd: np.random.randint(0, 1000, (3, 29)),
    #             page_phd: np.random.randint(0, 4, (3, 29))
    #         })
    #         print(o)
