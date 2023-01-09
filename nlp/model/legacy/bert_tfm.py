"""
https://arxiv.org/pdf/1910.00883.pdf
"""
from mesoorflow.layer import *
from mesoorflow.layer.activation import get_activation


class BertTFM(object):
    def __init__(self,
                 vocab_size: int = 21128,
                 type_vocab_size: int = 2,
                 embedding_dim: int = 768,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 attention_head_n: int = 12,
                 size_per_head: int = 64,
                 max_position: int = 512,
                 encoder_layer_n: int = 12,
                 dropout_rate: float = 0.2,
                 hidden_act: str = 'relu',
                 mask_token_id: int = 0,
                 position_embedding: str = 'learned',
                 **kwargs):
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_head_n = attention_head_n
        self.size_per_head = size_per_head
        self.max_position = max_position
        self.encoder_layer_n = encoder_layer_n

        self.hidden_act_fn = get_activation(hidden_act)
        self.hidden_act = hidden_act

        self.mask_token_id = mask_token_id
        self.is_training = False
        self.dropout_rate = dropout_rate

        self.crf_trans = None

        self.reuse = tf.AUTO_REUSE
        self.word_embedding_table = None
        self.position_embedding = position_embedding

        self.show_config()

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
        with tf.variable_scope("bert_tfm", reuse=self.reuse):
            with tf.variable_scope("embeddings", reuse=self.reuse):
                embedding_output, embedding_table = word_embedding(batch_token_ids, self.vocab_size, self.embedding_dim,
                                                                   get_new_initializer=self.get_new_initializer,
                                                                   name='word_embeddings')
                embedding_output = tf.Print(embedding_output, ['tfm/embedding', embedding_output], first_n=1, summarize=10)

                if batch_token_type_ids:
                    token_type_embedding_output, _ = word_embedding(batch_token_type_ids, self.type_vocab_size,
                                                                    self.embedding_dim,
                                                                    get_new_initializer=self.get_new_initializer,
                                                                    name='token_type_embeddings')
                    embedding_output += token_type_embedding_output
                if self.position_embedding == 'learned':
                    position_embedding_output, position_embedding_table = \
                        learned_position_embedding(batch_token_ids, self.max_position, self.embedding_dim,
                                                   get_new_initializer=self.get_new_initializer,
                                                   name='position_embeddings')

                    embedding_output = embedding_output + position_embedding_output
                # output = dropout(output, self.dropout_prob)
                self.word_embedding_table = embedding_table
        return embedding_output

    def encode(self, batch_token_ids: tf.Tensor, batch_token_type_ids: Optional[tf.Tensor] = None):
        batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)

        att_len_mask = length_mask(batch_token_ids, mask_value=self.mask_token_id)

        batch_embedded = self.embed(batch_token_ids, batch_token_type_ids)
        all_encoder_outputs = list()
        with tf.variable_scope("bert_tfm", reuse=self.reuse):
            with tf.variable_scope('embeddings', reuse=self.reuse):
                if self.position_embedding == 'relative':
                    position_embeddings = tf.get_variable("rel_position_embeddings",
                                                          [self.attention_head_n, 2 * self.max_position + 1],
                                                          tf.float32)
                    idx_matrix = tf.range(seq_len, dtype=tf.int32) - tf.expand_dims(
                        tf.range(seq_len, dtype=tf.int32), -1)
                    idx_matrix = tf.clip_by_value(idx_matrix, -self.max_position, self.max_position)
                    idx_matrix = idx_matrix + self.max_position  # seq_len,seq_len
                    idx_matrix = tf.tile(tf.expand_dims(idx_matrix, 0),
                                         [self.attention_head_n, 1, 1])  # head,seq_len,seq_len
                    head_idx = tf.tile(
                        tf.expand_dims(tf.expand_dims(tf.range(self.attention_head_n, dtype=tf.int64), -1), -1),
                        [1, seq_len, seq_len])  # head,seq_len,seq_len
                    idx_matrix = tf.stack([head_idx, idx_matrix], -1)
                    position_embeddings = tf.gather_nd(position_embeddings, idx_matrix)  # head,seq_len,seq_len
                    position_embeddings = tf.expand_dims(position_embeddings, 0)  # 1,head,seq_len,seq_len
                else:
                    position_embeddings = None
                batch_embedded = layer_norm(batch_embedded)
                batch_embedded = dropout(batch_embedded, self.dropout_prob)

            with tf.variable_scope("encoder", reuse=self.reuse):
                prev_output = batch_embedded
                for idx in range(self.encoder_layer_n):
                    layer_input = prev_output
                    if idx != 0:
                        att_input = self.hidden_act_fn(layer_norm(layer_input))
                    else:
                        att_input = layer_input
                    with tf.variable_scope(f"layer_{idx}"):
                        with tf.variable_scope("attention", reuse=self.reuse):

                            with tf.variable_scope("self"):
                                att_output, _, _ = multi_head_attention(query=att_input,
                                                                        attention_mask=att_len_mask,
                                                                        hidden_size=self.hidden_size,
                                                                        attention_head_n=self.attention_head_n,
                                                                        size_per_head=self.size_per_head,
                                                                        attention_probs_dropout_prob=self.dropout_prob,
                                                                        position_bias=position_embeddings,
                                                                        get_new_initializer=self.get_new_initializer)
                            with tf.variable_scope("output"):
                                att_output = tf.layers.dense(
                                    tf.reshape(att_output, [batch_size * seq_len, self.hidden_size]),
                                    units=self.hidden_size,
                                    kernel_initializer=self.get_new_initializer())
                                att_output = dropout(att_output, self.dropout_prob)
                                att_output = tf.reshape(att_output, shape=[batch_size, seq_len, self.hidden_size])
                            layer_output = att_output + layer_input
                            prev_output = layer_output
                            all_encoder_outputs.append(layer_output)
            final_output = layer_norm(all_encoder_outputs[-1])
        return final_output

    def masked_lm_outputs(self, batch_encoder_output: tf.Tensor, batch_label_weights: tf.Tensor,
                          batch_label_ids: tf.Tensor):
        batch_size, seq_len, _ = get_shape_list(batch_encoder_output, expected_rank=3)
        with tf.variable_scope("cls/predictions", reuse=self.reuse):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    tf.reshape(batch_encoder_output, shape=[batch_size * seq_len, self.hidden_size]),
                    units=self.hidden_size,
                    activation=self.hidden_act,
                    kernel_initializer=self.get_new_initializer())
                input_tensor = layer_norm(input_tensor)

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


if __name__ == '__main__':
    from tensorflow import data


    def model_fn(features, labels, mode):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        batch_token_ids = features["batch_token_ids"]

        model = BertTFM(vocab_size=21128, type_vocab_size=0, crf_label_n=2)

        batch_sequence_output = model.encode(batch_token_ids)
        loss, per_example_loss, log_probs = model.masked_lm_outputs(batch_sequence_output,
                                                                    tf.ones_like(batch_token_ids, dtype=tf.float32),
                                                                    batch_token_ids)
        # tf.train.init_from_checkpoint('./checkpoint/chinese_L-12_H-768_A-12/bert_model.ckpt',
        #                               {'bert/': 'bert/', 'cls/': 'cls/'})
        tf.global_variables_initializer()

        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                                     predictions={"loss": loss,
                                                                  "per_example_loss": per_example_loss,
                                                                  "log_probs": log_probs})
        else:
            raise ValueError("暂时只支持predict")

        return output_spec


    run_config = tf.estimator.RunConfig(
        log_step_count_steps=5000,
        tf_random_seed=19830610,
        save_checkpoints_steps=100,
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir='./output',
                                       config=run_config)


    def predict_input_fn():
        d = tf.data.Dataset.from_tensor_slices({
            "batch_token_ids":
                tf.constant(
                    [[[2, 8, 6544, 155, 1, 3, 25, 10, 11,
                       4634, 3339, 5, 39, 122, 13, 19, 1660, 13777,
                       5, 186, 12876, 7, 14, 19, 1660, 1277, 1406,
                       6, 1660, 956, 18, 12, 784, 254, 7, 16,
                       17780, 4088, 9, 36, 857, 505, 257, 71, 17780,
                       39, 1660, 9, 67, 4, 47, 10, 11, 290,
                       6204, 5, 8159, 5, 288, 7, 13, 10255, 5,
                       677, 7, 14, 213, 6648, 3817, 168, 5, 1925,
                       9, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0],
                      [2, 8, 5154, 1, 4, 35, 10, 11, 612,
                       12551, 7, 13, 61, 50, 45, 2273, 3344, 2688,
                       3908, 7142, 14, 5934, 5, 227, 5, 3830, 6,
                       4802, 7, 16, 20, 5154, 4197, 17246, 1091, 9,
                       3, 25, 10, 11, 1383, 33, 895, 19, 5,
                       58, 4326, 493, 1789, 438, 5, 13880, 5779, 5,
                       91, 209, 1800, 9, 13, 29, 3963, 6, 141,
                       3344, 308, 5, 210, 233, 293, 80, 4426, 5,
                       229, 13887, 5, 80, 173, 4094, 120, 3860, 5,
                       3014, 199, 3174, 5779, 15, 7222, 7, 14, 210,
                       134, 982, 6264, 506, 15, 224, 6264, 506, 7253,
                       17349, 12, 224, 6264, 506, 7, 16, 2078, 293,
                       220, 3492, 5, 9209, 4287, 220, 126, 6, 8385,
                       15, 2703, 5, 33, 220, 18144, 12, 2143, 19,
                       7, 22, 80, 14076, 15, 433, 2658, 6, 15857,
                       126, 4473, 5155, 524, 5, 80, 255, 7670, 166,
                       7393, 17, 7699, 5, 190, 9509, 217, 11513, 6,
                       3881, 15, 130, 16667, 243, 7, 36, 4906, 120,
                       6584, 15, 2143, 38, 433, 15, 1227, 393, 7,
                       67, 588, 17059, 978, 3276, 5, 6355, 7871, 2941,
                       9, 176, 7433, 48, 5364, 9, 1]]], shape=[1, 2, 187],
                    dtype=tf.int32),
            "batch_seq_label":
                tf.constant(
                    [
                        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                    ], shape=[1, 2, 187],
                    dtype=tf.int32
                ),
        })

        return d.repeat(1)


    outputs = list()
    for output in estimator.predict(predict_input_fn, yield_single_examples=False):
        print('predict')
        for k, v in output.items():
            print(k)
            print(v.shape)
            # print(v.argmax(axis=-1))
            if k == 'batch_log_probs':
                print(v[:, :5, :])
            else:
                print(v)
