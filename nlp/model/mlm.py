from typing import Optional, Dict

import tensorflow as tf
from pydantic import validator

from nlp.layer.utils import get_shape_list, assert_rank
from nlp.loss import mlm_loss
from nlp.model.model import Model


@Model.register("mlm", exist_ok=True)
class MLM(Model):
    encoder: Model
    l2_normalize: bool = True
    need_bias: bool = True

    @validator("encoder", pre=True)
    def load_encoder(cls, encoder_params):
        return Model.from_params(**encoder_params)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        from nlp.model.bert import Bert
        if isinstance(self.encoder, Bert):
            encoder_output = self.encoder.forward(inputs)
            batch_encoder_out = encoder_output['batch_encoder_out']
            word_embedding_table = encoder_output['embedding_table']
        else:
            raise ValueError(f"{self.__class__} 没有适配encoder: {self.encoder.__class__}")

        batch_size, seq_len, hidden_dim = get_shape_list(batch_encoder_out, expected_rank=3)
        vocab_size, embedding_dim = get_shape_list(word_embedding_table, expected_rank=2)

        with tf.variable_scope("mlm", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    tf.reshape(batch_encoder_out, shape=[batch_size * seq_len, hidden_dim]),
                    units=embedding_dim,
                    activation=None,
                    kernel_initializer=self.encoder.get_new_initializer())
                if self.l2_normalize:
                    input_tensor = tf.nn.l2_normalize(input_tensor, axis=-1)

            # bs * seq_len, vocab_size
            batch_logit = tf.matmul(input_tensor, word_embedding_table, transpose_b=True)

            if self.need_bias:
                output_bias = tf.get_variable("output_bias",
                                              shape=[vocab_size],
                                              initializer=tf.zeros_initializer())
                batch_logit = tf.nn.bias_add(batch_logit, output_bias)

            batch_logit = tf.reshape(batch_logit, shape=[batch_size, seq_len, vocab_size])
        return {"batch_logit": batch_logit}

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        batch_logit = outputs["batch_logit"]
        batch_label_ids = labels["batch_label_ids"]
        assert weights is not None
        batch_weights = weights["batch_label_weights"]
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            loss = mlm_loss(batch_logit, batch_label_ids, batch_weights)
        return loss

    #
    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        batch_logit = outputs["batch_logit"]
        batch_label_ids = labels["batch_label_ids"]
        assert weights is not None
        batch_weights = weights["batch_label_weights"]
        assert_rank(batch_logit, expected_rank=3)
        assert_rank(batch_label_ids, expected_rank=2)
        assert_rank(batch_weights, expected_rank=2)
        batch_prediction = tf.argmax(batch_logit, axis=-1)
        acc = tf.metrics.accuracy(labels=batch_label_ids,
                                  predictions=batch_prediction,
                                  weights=batch_weights)
        recall = tf.metrics.recall(labels=batch_label_ids,
                                   predictions=batch_prediction,
                                   weights=batch_weights)
        return {"acc": acc, "recall": recall}
