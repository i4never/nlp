from typing import List

from mesoorflow.layer import *
from mesoorflow.layer.seq_tagger import isolated_span_outputs


class Bert(object):
    def __init__(self,
                 vocab_size: int = 21128,
                 type_vocab_size: int = 2,
                 embedding_dim: int = 768,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 attention_head_n: int = 12,
                 size_per_head: int = 64,
                 position_embedding_type: str = 'trained',
                 max_position: int = 512,
                 encoder_layer_n: int = 12,
                 dropout_rate: float = 0.2,
                 hidden_act: str = 'gelu',
                 mask_token_id: int = 0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_head_n = attention_head_n
        self.size_per_head = size_per_head
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type not in ['trained', 'relative']:
            raise ValueError(f"不支持位置编码 {self.position_embedding_type}")
        self.max_position = max_position
        self.encoder_layer_n = encoder_layer_n
        self.hidden_act = get_activation(hidden_act)
        self.mask_token_id = mask_token_id
        self.is_training = False
        self.dropout_rate = dropout_rate

        self.reuse = tf.AUTO_REUSE
        self.word_embedding_table = None

        self.attention_length_mask = None
        self.position_bias = None
        self.show_config()

    def inputs(self):
        return {
            "batch_token_ids": tf.placeholder(tf.int32, [None, None], name="batch_token_ids"),
            "batch_token_type_ids": tf.placeholder(tf.int32, [None, None], name="batch_token_type_ids")
        }

    def show_config(self):
        for k, v in self.__dict__.items():
            print(f'{k}={v}')
        print()

    def get_new_initializer(self, stddev: float = 0.02):
        return tf.truncated_normal_initializer(stddev=stddev)

    @property
    def dropout_prob(self):
        return self.dropout_rate if self.is_training else 0.

    def embed(self, batch_token_ids: tf.Tensor, batch_token_type_ids: Optional[tf.Tensor] = None):
        with tf.variable_scope("bert", reuse=self.reuse):
            with tf.variable_scope("embeddings", reuse=self.reuse):
                embedding_output, embedding_table = word_embedding(batch_token_ids, self.vocab_size, self.embedding_dim,
                                                                   get_new_initializer=self.get_new_initializer,
                                                                   name='word_embeddings')
                embedding_output = tf.Print(embedding_output, ['bert/embedding', embedding_output], first_n=1,
                                            summarize=10)

                if batch_token_type_ids:
                    token_type_embedding_output, _ = word_embedding(batch_token_type_ids, self.type_vocab_size,
                                                                    self.embedding_dim,
                                                                    get_new_initializer=self.get_new_initializer,
                                                                    name='token_type_embeddings')
                    embedding_output += token_type_embedding_output

                if self.position_embedding_type == 'trained':
                    position_embedding_output, position_embedding_table = \
                        learned_position_embedding(batch_token_ids, self.max_position, self.embedding_dim,
                                                   get_new_initializer=self.get_new_initializer,
                                                   name='position_embeddings')
                    embedding_output += position_embedding_output

                output = layer_norm(embedding_output, 'norm_embedding')
                output = dropout(output, self.dropout_prob)
                self.word_embedding_table = embedding_table
        return output

    def encode(self, batch_token_ids: tf.Tensor, batch_token_type_ids: Optional[tf.Tensor] = None,
               attention_mask: Optional[tf.Tensor] = None):
        batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)

        self.attention_length_mask = length_mask(batch_token_ids, mask_value=self.mask_token_id)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32) * tf.cast(self.attention_length_mask, tf.float32)
        else:
            attention_mask = self.attention_length_mask

        batch_embedded = self.embed(batch_token_ids, batch_token_type_ids)
        all_encoder_outputs = list()
        with tf.variable_scope("bert", reuse=self.reuse):
            if self.position_embedding_type == 'relative':
                position_bias, _ = learned_relative_position_embedding(seq_len, seq_len, self.max_position, 1)
                position_bias = tf.squeeze(position_bias, -1)
                self.position_bias = tf.repeat(tf.expand_dims(position_bias, 0), batch_size, axis=0)
            else:
                self.position_bias = None

            with tf.variable_scope("encoder", reuse=self.reuse):
                prev_output = batch_embedded
                for idx in range(self.encoder_layer_n):
                    with tf.variable_scope(f"layer_{idx}"):
                        with tf.variable_scope("attention", reuse=self.reuse):
                            with tf.variable_scope("self"):
                                att_output, _, _ = multi_head_attention(query=prev_output,
                                                                        attention_mask=attention_mask,
                                                                        position_bias=self.position_bias,
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
                                                                      activation=self.hidden_act,
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
        return all_encoder_outputs[-1]

    def token_cls_outputs(self, batch_encoder_output, class_n, activation=None,
                          get_new_initializer_foo=lambda stddev=0.2: tf.truncated_normal_initializer(stddev=stddev)):
        batch_size, seq_len, hidden_size = get_shape_list(batch_encoder_output, expected_rank=3)
        with tf.variable_scope("cls/tokens", reuse=self.reuse):
            with tf.variable_scope("transform"):
                output = tf.layers.dense(
                    tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, hidden_size]),
                    activation=activation,
                    units=class_n,
                    kernel_initializer=get_new_initializer_foo()
                )
                output = tf.reshape(output, shape=[batch_size, seq_len, class_n])
        return output

    def masked_lm_outputs(self, batch_encoder_output: tf.Tensor, batch_label_weights: tf.Tensor,
                          batch_label_ids: tf.Tensor):
        batch_size, seq_len, _ = get_shape_list(batch_encoder_output, expected_rank=3)
        with tf.variable_scope("cls/predictions", reuse=self.reuse):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, self.hidden_size]),
                    units=self.hidden_size,
                    activation=None,
                    kernel_initializer=self.get_new_initializer())
                input_tensor = tf.nn.l2_normalize(input_tensor, axis=-1)

            output_bias = tf.get_variable("output_bias",
                                          shape=[self.vocab_size],
                                          initializer=tf.zeros_initializer())

            batch_logits = tf.matmul(input_tensor, self.word_embedding_table, transpose_b=True)
            batch_logits = tf.nn.bias_add(batch_logits, output_bias)
            batch_log_probs = tf.nn.log_softmax(batch_logits, axis=-1)

            batch_label_ids = tf.reshape(batch_label_ids, [-1])
            batch_label_weights = tf.reshape(batch_label_weights, [-1])

            batch_one_hot_labels = tf.one_hot(batch_label_ids, depth=self.vocab_size, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(batch_log_probs * batch_one_hot_labels, axis=-1)
            numerator = tf.reduce_sum(batch_label_weights * per_example_loss)
            denominator = tf.reduce_sum(batch_label_weights) + 1e-5
            loss = numerator / denominator
            batch_log_probs = tf.reshape(batch_log_probs, shape=[batch_size, seq_len, -1])
            return loss, per_example_loss, batch_log_probs

    def span_outputs(self, entities: List[str], batch_encoder_output: tf.Tensor) -> List[List[tf.Tensor]]:
        return isolated_span_outputs(entities, batch_encoder_output, self.get_new_initializer)

    def pooler(self, batch_encoder_output: tf.Tensor, batch_token_mask: Optional[tf.Tensor] = None,
               method: str = 'mean', large_negative_number: float = tf.float32.min):
        assert_rank(batch_encoder_output, expected_rank=3)
        assert_rank(batch_token_mask, expected_rank=2)
        if method not in ['cls', 'mean', 'max']:
            raise ValueError(f"Unknown pooler method {method}")

        if method == 'cls':
            return batch_encoder_output[:, 0, :]

        if method == 'mean':
            x = batch_encoder_output * batch_token_mask[:, :, None]
            x = tf.reduce_sum(x, axis=1)
            x = x * (1 / tf.reduce_sum(batch_token_mask, axis=-1, keepdims=True))
            return x

        if method == 'max':
            x = batch_encoder_output + (1. - batch_token_mask[:, :, None]) * large_negative_number
            x = tf.reduce_max(x, axis=1)
            return x
