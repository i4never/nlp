from nlp.layer import *


class Bart(object):
    def __init__(self,
                 vocab_size: int = 21128,
                 embedding_dim: int = 768,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 attention_head_n: int = 12,
                 size_per_head: int = 64,
                 max_position: int = 512,
                 encoder_layer_n: int = 6,
                 decoder_layer_n: int = 6,
                 dropout_rate: float = 0.2,
                 hidden_act: str ='gelu',
                 mask_token_id: int = 0):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_head_n = attention_head_n
        self.size_per_head = size_per_head
        self.max_position = max_position
        self.encoder_layer_n = encoder_layer_n
        self.decoder_layer_n = decoder_layer_n
        if hidden_act == 'gelu':
            self.hidden_act = gelu
        elif hidden_act == 'relu':
            self.hidden_act = 'relu'
        else:
            raise ValueError('only gelu act for now')
        self.mask_token_id = mask_token_id
        self.is_training = False
        self.dropout_rate = dropout_rate

        self.reuse = tf.AUTO_REUSE
        self.word_embedding_table = None

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

    def embed(self, batch_token_ids: tf.Tensor):
        with tf.variable_scope("bart", reuse=self.reuse):
            with tf.variable_scope("embeddings", reuse=self.reuse):
                embedding_output, embedding_table = word_embedding(batch_token_ids, self.vocab_size, self.embedding_dim,
                                                                   get_new_initializer=self.get_new_initializer,
                                                                   name='word_embeddings')

                position_embedding_output, position_embedding_table = \
                    learned_position_embedding(batch_token_ids, self.max_position, self.embedding_dim,
                                               get_new_initializer=self.get_new_initializer,
                                               name='learned_position_embeddings')

                output = embedding_output + position_embedding_output
                output = layer_norm(output, 'norm_embedding')
                output = dropout(output, self.dropout_prob)
                self.word_embedding_table = embedding_table
        return output

    def encode(self, batch_token_ids: tf.Tensor):
        batch_size, seq_len = get_shape_list(batch_token_ids, expected_rank=2)
        embedded = self.embed(batch_token_ids)  # TODO: * embed_scale?
        att_len_mask = length_mask(batch_token_ids, self.mask_token_id)
        # att_len_mask = tf.Print(att_len_mask, ['att_len_mask', att_len_mask], summarize=100)

        all_encode_outputs = list()
        with tf.variable_scope("encoder", reuse=self.reuse):
            prev_output = embedded
            for idx in range(self.encoder_layer_n):
                with tf.variable_scope(f"layer_{idx}", reuse=self.reuse):
                    att_output, att_weights, _ = multi_head_attention(query=prev_output,
                                                                      attention_mask=att_len_mask,
                                                                      hidden_size=self.hidden_size,
                                                                      attention_head_n=self.attention_head_n,
                                                                      size_per_head=self.size_per_head,
                                                                      attention_probs_dropout_prob=self.dropout_prob,
                                                                      get_new_initializer=self.get_new_initializer)
                    # batch_size, seq_len, hidden_size
                    hidden_state = dropout(att_output, dropout_prob=self.dropout_prob)
                    hidden_state = prev_output + hidden_state
                    hidden_state = layer_norm(hidden_state)

                    fc_input = hidden_state
                    hidden_state = tf.layers.dense(tf.reshape(hidden_state, [batch_size * seq_len, self.hidden_size]),
                                                   units=self.intermediate_size,
                                                   activation=self.hidden_act,
                                                   name="fc1",
                                                   kernel_initializer=self.get_new_initializer())
                    hidden_state = dropout(hidden_state, dropout_prob=self.dropout_prob)
                    hidden_state = tf.layers.dense(hidden_state,
                                                   units=self.hidden_size,
                                                   activation=self.hidden_act,
                                                   name="fc2",
                                                   kernel_initializer=self.get_new_initializer())
                    hidden_state = dropout(hidden_state, dropout_prob=self.dropout_prob)
                    hidden_state = tf.reshape(hidden_state, shape=[batch_size, seq_len, self.hidden_size])
                    hidden_state = fc_input + hidden_state
                    hidden_state = layer_norm(hidden_state)
                    prev_output = hidden_state
                    all_encode_outputs.append(hidden_state)

        return all_encode_outputs[-1]

    def decode(self,
               batch_encoder_token_ids: tf.Tensor,
               batch_decoder_token_ids: tf.Tensor,
               batch_encoder_outputs: tf.Tensor,
               batch_self_att_cache_key_value: Optional[tf.Tensor] = None,
               batch_cross_att_cache_key_value: Optional[tf.Tensor] = None) -> Tuple:
        batch_size, decoder_seq_len = get_shape_list(batch_decoder_token_ids, expected_rank=2)
        reuse = self.reuse
        self.reuse = True
        decoder_embedded = self.embed(batch_decoder_token_ids)  # TODO: * embed_scale?
        self.reuse = reuse

        # decoder self att mask
        decoder_len_mask = length_mask(batch_decoder_token_ids, self.mask_token_id)
        decoder_lm_mask = ar_lm_mask(batch_decoder_token_ids)
        decoder_self_att_mask = decoder_len_mask * decoder_lm_mask

        # decoder_self_att_mask = tf.Print(decoder_self_att_mask, ['decoder_self_att_mask', decoder_self_att_mask],
        #                                  summarize=100, first_n=10)

        # decoder cross att mask
        encoder_mask = tf.cast(tf.not_equal(batch_encoder_token_ids, self.mask_token_id), dtype=tf.float32)
        decoder_cross_att_mask = tf.tile(encoder_mask[:, None, :], (1, decoder_seq_len, 1))

        assert not ((batch_self_att_cache_key_value is not None) ^ (batch_cross_att_cache_key_value is not None))
        if batch_self_att_cache_key_value is not None:
            batch_self_att_cache_key_value = tf.unstack(batch_self_att_cache_key_value, axis=0)
            batch_self_att_cache_key_value_tuples = list()
            for idx in range(self.decoder_layer_n):
                unstacked = tf.unstack(batch_self_att_cache_key_value[idx], axis=0)
                batch_self_att_cache_key_value_tuples.append((unstacked[0], unstacked[1]))

            batch_cross_att_cache_key_value = tf.unstack(batch_cross_att_cache_key_value, axis=0)
            batch_cross_att_cache_key_value_tuples = list()
            for idx in range(self.decoder_layer_n):
                unstacked = tf.unstack(batch_cross_att_cache_key_value[idx], axis=0)
                batch_cross_att_cache_key_value_tuples.append((unstacked[0], unstacked[1]))

            decoder_embedded = decoder_embedded[:, -1:, :]
            decoder_seq_len = 1
            decoder_self_att_mask = decoder_self_att_mask[:, -1:, :]
            decoder_cross_att_mask = decoder_cross_att_mask[:, -1:, :]
        else:
            batch_self_att_cache_key_value_tuples = None
            batch_cross_att_cache_key_value_tuples = None

        all_hidden_states = list()
        present_self_att_key_values = list()
        present_cross_att_key_values = list()
        with tf.variable_scope("decoder", reuse=self.reuse):
            prev_output = decoder_embedded
            for idx in range(self.decoder_layer_n):
                with tf.variable_scope(f"layer_{idx}", reuse=self.reuse):
                    with tf.variable_scope("self", reuse=self.reuse):
                        self_att_output, _, present_self_att_key_value = \
                            multi_head_attention(query=prev_output,
                                                 cache_key_value_states=batch_self_att_cache_key_value_tuples[
                                                     idx] if batch_self_att_cache_key_value_tuples is not None else None,
                                                 attention_mask=decoder_self_att_mask,
                                                 hidden_size=self.hidden_size,
                                                 attention_head_n=self.attention_head_n,
                                                 size_per_head=self.size_per_head,
                                                 attention_probs_dropout_prob=self.dropout_prob,
                                                 get_new_initializer=self.get_new_initializer)
                        present_self_att_key_values.append(present_self_att_key_value)
                        hidden_state = dropout(self_att_output, self.dropout_prob)
                        hidden_state = prev_output + hidden_state
                        hidden_state = layer_norm(hidden_state)

                    with tf.variable_scope("cross", reuse=self.reuse):
                        cross_input = hidden_state
                        cross_att_output, _, present_cross_att_key_value = \
                            multi_head_attention(query=hidden_state,
                                                 key_value=batch_encoder_outputs,
                                                 cache_key_value_states=batch_cross_att_cache_key_value_tuples[
                                                     idx] if batch_cross_att_cache_key_value_tuples is not None else None,
                                                 attention_mask=decoder_cross_att_mask,
                                                 hidden_size=self.hidden_size,
                                                 attention_head_n=self.attention_head_n,
                                                 size_per_head=self.size_per_head,
                                                 attention_probs_dropout_prob=self.dropout_prob,
                                                 get_new_initializer=self.get_new_initializer)
                        present_cross_att_key_values.append(present_cross_att_key_value)
                        hidden_state = dropout(cross_att_output, self.dropout_prob)
                        hidden_state = cross_input + hidden_state
                        hidden_state = layer_norm(hidden_state)

                    fc_input = hidden_state
                    hidden_state = tf.layers.dense(
                        tf.reshape(hidden_state, [batch_size * decoder_seq_len, self.hidden_size]),
                        units=self.intermediate_size,
                        activation=self.hidden_act,
                        name="fc1",
                        kernel_initializer=self.get_new_initializer())
                    hidden_state = dropout(hidden_state, dropout_prob=self.dropout_prob)
                    hidden_state = tf.layers.dense(hidden_state,
                                                   units=self.hidden_size,
                                                   activation=self.hidden_act,
                                                   name="fc2",
                                                   kernel_initializer=self.get_new_initializer())
                    hidden_state = dropout(hidden_state, dropout_prob=self.dropout_prob)
                    hidden_state = tf.reshape(hidden_state, shape=[batch_size, decoder_seq_len, self.hidden_size])
                    hidden_state = fc_input + hidden_state
                    hidden_state = layer_norm(hidden_state)

                    prev_output = hidden_state
                    all_hidden_states.append(hidden_state)

        return (
            all_hidden_states[-1],
            tf.stack([tf.stack(c, axis=0) for c in present_self_att_key_values], axis=0),
            tf.stack([tf.stack(c, axis=0) for c in present_cross_att_key_values], axis=0)
        )

    def lm_output(self, batch_decoder_output: tf.Tensor):
        return tf.matmul(batch_decoder_output, self.word_embedding_table, transpose_b=True)

    def lm_loss(self, batch_decoder_output: tf.Tensor, batch_decoder_token_ids: tf.Tensor):
        logits = self.lm_output(batch_decoder_output)
        mask = tf.not_equal(batch_decoder_token_ids, self.mask_token_id)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.losses.sparse_softmax_cross_entropy(batch_decoder_token_ids[:, 1:], logits[:, :-1, :],
                                                      weights=mask[:, 1:])


if __name__ == "__main__":
    def model_fn(features, labels, mode):

        bart = Bart(encoder_layer_n=6)

        batch_encoder_token_ids = features['batch_encoder_token_ids']
        batch_decoder_token_ids = features['batch_decoder_token_ids']

        batch_encoder_outputs = bart.encode(batch_encoder_token_ids)

        batch_decoder_output, batch_cache_kvs = bart.decode(batch_encoder_token_ids,
                                                            batch_decoder_token_ids,
                                                            batch_encoder_outputs)

        loss = bart.lm_loss(batch_decoder_output, batch_decoder_token_ids)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # for v in tf.trainable_variables():
            #     print(v.name)
            # tf.train.init_from_checkpoint('./checkpoint/chinese_L-12_H-768_A-12/bert_model.ckpt',
            #                               {'bert/': 'bert/'})
            tf.global_variables_initializer()
            output_spec = tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                                     predictions={"batch_encoder_outputs": batch_encoder_outputs,
                                                                  "batch_decoder_output": batch_decoder_output,
                                                                  "loss": loss})
        return output_spec


    def gen():
        yield {
            "batch_encoder_token_ids": np.array([
                [10, 12, 1294, 124, 4534, 0, 0],
                [123, 123, 7, 34, 2, 1, 0]]),
            "batch_decoder_token_ids": np.array([
                [10, 12, 1294, 124, 4534, 0],
                [123, 123, 7, 34, 2, 0]
            ])}
        yield {
            "batch_encoder_token_ids": np.array([
                [10, 12, 1294, 124, 4534, 0, 0, 0, 0, 0, 0],
                [123, 123, 7, 34, 2, 1, 0, 0, 0, 0, 0]
            ]),
            "batch_decoder_token_ids": np.array([
                [10, 12, 1294, 124, 4534, 0, 0],
                [123, 123, 7, 34, 2, 0, 0]
            ])}


    def predict_input_fn():
        d = tf.data.Dataset.from_generator(
            gen,
            output_types={"batch_encoder_token_ids": tf.int32, "batch_decoder_token_ids": tf.int32},
            output_shapes={"batch_encoder_token_ids": [None, None], "batch_decoder_token_ids": [None, None]}
        )
        return d.repeat(1)


    run_config = tf.estimator.RunConfig(
        log_step_count_steps=5000,
        tf_random_seed=19830610,
        save_checkpoints_steps=100,
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir='./output',
                                       config=run_config)
    for output in estimator.predict(predict_input_fn, yield_single_examples=False):
        for k, v in output.items():
            print(k, v.shape)
            print(v)
        print("")
