import tensorflow as tf
from pydantic import validator
from typing import Optional, Dict, Any, List, Tuple, Union

from nlp.layer.attention import multi_head_attention
from nlp.layer.dropout import dropout
from nlp.layer.embedding import word_embedding, learned_position_embedding, learned_relative_position_embedding, \
    PositionEmbeddingType
from nlp.layer.mask import length_mask
from nlp.layer.norm import layer_norm
from nlp.layer.pooler import first_token_pooler, sequence_mean_pooler, sequence_max_pooler, PoolerType
from nlp.layer.seq_tagger import crf_outputs
from nlp.layer.utils import get_shape_list, assert_rank
from nlp.layer.activation import get_activation
from nlp.loss import mlm_loss
from nlp.model.model import Model
import numpy as np


@Model.register("bert", exist_ok=True)
class Bert(Model):
    vocab_size: int = 21128
    type_vocab_size: int = 2
    embedding_dim: int = 768
    hidden_size: int = 768
    intermediate_size: int = 3072
    attention_head_n: int = 12
    size_per_head: int = 64
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_fixed
    max_position: int = 512  # relative: 前后相对长度 fixed: 最大长度
    encoder_layer_n: int = 12
    dropout_rate: float = 0.2
    hidden_act: str = 'gelu'
    pad_token_id: int = 0
    normal_initializer_stddev: float = 0.02
    pooler: Optional[PoolerType] = None

    def __init__(self, **kwargs):
        super(Bert, self).__init__(**kwargs)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """

        :param inputs: (batch_token_ids,) or (batch_token_ids, batch_token_type_ids)
        :return: (最后一层pooled输出，所有block输出，embedding lookup表)
        """
        if self.type_vocab_size < 2:
            batch_token_ids = inputs['batch_token_ids']
            batch_token_type_ids = None
        else:
            batch_token_ids = inputs['batch_token_type_ids']
            batch_token_type_ids = inputs['batch_token_type_ids']

        with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):
            batch_length_mask = length_mask(batch_token_ids, pad_value=self.pad_token_id)
            with tf.variable_scope("embeddings"):
                batch_embedded, embedding_table = self.embed(batch_token_ids, batch_token_type_ids)
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

    def embed(self, batch_token_ids: tf.Tensor, batch_token_type_ids: Optional[tf.Tensor] = None) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        embedding_output, embedding_table = word_embedding(batch_token_ids, self.vocab_size, self.embedding_dim,
                                                           get_new_initializer=self.get_new_initializer,
                                                           name='word_embeddings')

        if batch_token_type_ids:
            token_type_embedding_output, _ = word_embedding(batch_token_type_ids, self.type_vocab_size,
                                                            self.embedding_dim,
                                                            get_new_initializer=self.get_new_initializer,
                                                            name='token_type_embeddings')
            embedding_output += token_type_embedding_output

        if self.position_embedding_type == 'fixed':
            position_embedding_output, position_embedding_table = \
                learned_position_embedding(batch_token_ids, self.max_position, self.embedding_dim,
                                           get_new_initializer=self.get_new_initializer,
                                           name='position_embeddings')
            embedding_output += position_embedding_output

        output = layer_norm(embedding_output, 'norm_embedding')
        output = dropout(output, self.dropout_prob)
        return output, embedding_table

    def encode(self, batch_embedded: tf.Tensor, batch_length_mask: tf.Tensor,
               attention_mask: Optional[tf.Tensor] = None) -> List[tf.Tensor]:
        batch_size, seq_len, _ = get_shape_list(batch_embedded, expected_rank=3)

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32) * tf.cast(batch_length_mask, tf.float32)
        else:
            attention_mask = batch_length_mask

        all_encoder_outputs = list()
        if self.position_embedding_type == 'relative':
            position_bias, _ = learned_relative_position_embedding(seq_len, seq_len, self.max_position, 1)
            position_bias = tf.squeeze(position_bias, -1)
            position_bias = tf.repeat(tf.expand_dims(position_bias, 0), batch_size, axis=0)
        else:
            position_bias = None

        prev_output = batch_embedded
        for idx in range(self.encoder_layer_n):
            with tf.variable_scope(f"layer_{idx}"):
                with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                    with tf.variable_scope("self"):
                        att_output, _, _ = multi_head_attention(query=prev_output,
                                                                attention_mask=attention_mask,
                                                                position_bias=position_bias,
                                                                hidden_size=self.hidden_size,
                                                                attention_head_n=self.attention_head_n,
                                                                size_per_head=self.size_per_head,
                                                                attention_probs_dropout_prob=self.dropout_prob,
                                                                get_new_initializer=self.get_new_initializer)
                    with tf.variable_scope("attention_output"):
                        att_output = tf.layers.dense(
                            tf.reshape(att_output, [batch_size * seq_len, self.hidden_size]),
                            units=self.hidden_size,
                            kernel_initializer=self.get_new_initializer())
                        att_output = dropout(att_output, self.dropout_prob)
                        att_output = layer_norm(att_output + tf.reshape(prev_output,
                                                                        shape=[batch_size * seq_len,
                                                                               self.hidden_size]))

                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.layers.dense(att_output,
                                                              units=self.intermediate_size,
                                                              activation=get_activation(self.hidden_act),
                                                              kernel_initializer=self.get_new_initializer())
                    with tf.variable_scope("intermediate_output"):
                        layer_output = tf.layers.dense(intermediate_output,
                                                       units=self.hidden_size,
                                                       kernel_initializer=self.get_new_initializer())
                        layer_output = dropout(layer_output, self.dropout_prob)
                        layer_output = layer_norm(layer_output + att_output)
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
        raise NotImplementedError(f"bert只包含encoder部分，没有具体任务定义，需要使用MLM或者其他任务加载bert模型")

    def _to_params(self) -> Dict[str, Any]:
        params = self.dict()
        params['type'] = 'bert'
        return params


if __name__ == '__main__':
    import numpy as np
    from nlp.common.registrable import import_all_modules_for_register
    import_all_modules_for_register()
    crf = Model.from_params(
        **{"encoder": {"name": "bert", "type_vocab_size": 0, "dropout_rate": 0}, "name": "crf_ner", "label_n": 9,
           "dropout_rate": 0.2})
    print(crf)

    input_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    label_phd = tf.placeholder(shape=(None, None), dtype=tf.int32)
    length_phd = tf.placeholder(shape=(None,), dtype=tf.int32)

    out = crf.forward({"batch_token_ids": input_phd})
    loss = crf.loss(out, {"batch_token_labels": label_phd, "batch_sequence_length": length_phd}, None)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        for _ in range(10):
            ids = np.random.randint(0, 1000, (5, 27))
            target_seq = np.random.randint(0, 9, (5, 27))
            length = np.random.randint(1, 27, (5,))
            o = sess.run([out, loss], feed_dict={input_phd: ids,
                                                 label_phd: target_seq,
                                                 length_phd: length})
            print(o)
