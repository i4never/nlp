from typing import Optional, Dict

import numpy as np
import tensorflow as tf
from pydantic import validator

from nlp.layer import length_mask
from nlp.layer.seq_tagger import crf_outputs, global_pointer
from nlp.loss import multilabel_categorical_crossentropy
from nlp.model.model import Model


@Model.register("crf_ner", exist_ok=True)
class CrfNer(Model):
    label_n: int
    encoder: Model
    l2_normalize: bool = True
    initialize_trans: bool = False  # 0为O、奇数为B-*、偶数为I-*，非法转移位置预设权重惩罚

    class Config:
        arbitrary_types_allowed = True

    @validator("encoder", pre=True)
    def load_encoder(cls, encoder_params):
        return Model.from_params(**encoder_params)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        from nlp.model.bert import Bert
        from nlp.model.bert_linear import Bert2DLinear
        from nlp.model.rec_bert_linear import RecBertLinear
        if type(self.encoder) in [Bert, Bert2DLinear, RecBertLinear]:
            encoder_output = self.encoder.forward(inputs)
            batch_encoder_out = encoder_output['batch_encoder_out']
            batch_length_mask = tf.cast(tf.not_equal(inputs["batch_token_ids"], 0), tf.int32)
            batch_length_mask = tf.identity(batch_length_mask, name="batch_length_mask")
            batch_length = tf.reduce_sum(batch_length_mask, axis=-1)
            batch_length = tf.identity(batch_length, name="batch_length")
        else:
            raise ValueError(f"{self.__class__} 没有适配encoder: {self.encoder.__class__}")

        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            if self.initialize_trans:
                trans_bias = np.random.normal(size=[self.label_n, self.label_n]).astype(np.float32)
                for i in range(self.label_n):
                    for j in range(self.label_n):
                        if i == 0 and j != 0 and j % 2 == 0:  # O转I
                            trans_bias[i, j] = -1e2
                        elif i % 2 == 1 and j % 2 == 0 and j != i + 1:  # B-a 转 I-b
                            trans_bias[i, j] = -1e2
                        elif i % 2 == 0 and j % 2 == 0 and i != j:  # I-a 转 I-b
                            trans_bias[i, j] = -1e2
            else:
                trans_bias = None
            batch_token_labels, crf_trans = crf_outputs(batch_encoder_out,
                                                        self.label_n,
                                                        activation=None,
                                                        trans=trans_bias,
                                                        get_new_initializer_foo=self.encoder.get_new_initializer)
            batch_token_labels = tf.identity(batch_token_labels, name='batch_token_labels')
            batch_viterbi_decoded_labels, _ = tf.contrib.crf.crf_decode(batch_token_labels,
                                                                        crf_trans,
                                                                        batch_length)
            batch_viterbi_decoded_labels = tf.identity(batch_viterbi_decoded_labels,
                                                       name='batch_viterbi_decoded_labels')
        return {
            "batch_token_labels": batch_token_labels,
            "batch_lengths": batch_length,
            "crf_trans": crf_trans,
            "batch_viterbi_decoded_labels": batch_viterbi_decoded_labels,
            "batch_length_mask": batch_length_mask
        }

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        batch_labels = tf.identity(labels["batch_token_labels"], "batch_token_labels")
        print(batch_labels)
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(outputs["batch_token_labels"],
                                                                  batch_labels,
                                                                  tf.reshape(outputs["batch_lengths"], [-1]),
                                                                  outputs["crf_trans"])
            loss = tf.reduce_mean(-log_likelihood)
        return loss

    #
    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        return dict()
        # batch_predict = tf.argmax(outputs["batch_token_labels"], axis=-1)
        # return {
        #     "acc": tf.metrics.accuracy(predictions=batch_predict,
        #                                labels=labels["batch_token_labels"],
        #                                weights=tf.cast(outputs["batch_length_mask"], tf.float32)),
        #     "recall": tf.metrics.recall(predictions=batch_predict,
        #                                 labels=labels["batch_token_labels"],
        #                                 weights=tf.cast(outputs["batch_length_mask"], tf.float32))
        # }


@Model.register("global_pointer", exist_ok=True)
class GlobalPointer(Model):
    label_n: int
    encoder: Model
    l2_normalize: bool = True

    class Config:
        arbitrary_types_allowed = True

    @validator("encoder", pre=True)
    def load_encoder(cls, encoder_params):
        return Model.from_params(**encoder_params)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        from nlp.model.bert import Bert
        from nlp.model.bert_linear import Bert2DLinear
        if type(self.encoder) in [Bert, Bert2DLinear]:
            encoder_output = self.encoder.forward(inputs)
            batch_encoder_out = encoder_output['batch_encoder_out']
        else:
            raise ValueError(f"{self.__class__} 没有适配encoder: {self.encoder.__class__}")

        with tf.variable_scope("global_pointer", reuse=tf.AUTO_REUSE):
            att_mask = length_mask(inputs["batch_token_ids"])
            att_weights, tri_mask = global_pointer(batch_encoder_out,
                                                   entity_n=self.label_n,
                                                   size_per_head=self.encoder.size_per_head,
                                                   attention_mask=att_mask)
        return {"batch_global_pointer": att_weights, "batch_tri_mask": tri_mask}

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            loss = multilabel_categorical_crossentropy(outputs["batch_global_pointer"],
                                                       labels["batch_global_pointer"],
                                                       weight=outputs["batch_tri_mask"])
        return loss

    #
    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        batch_predict = tf.cast(outputs["batch_global_pointer"] > 0, tf.float32)
        return {"acc": tf.metrics.accuracy(predictions=batch_predict,
                                           labels=labels["batch_global_pointer"],
                                           weights=outputs["batch_tri_mask"]),
                "recall": tf.metrics.recall(predictions=batch_predict,
                                            labels=labels["batch_global_pointer"],
                                            weights=outputs["batch_tri_mask"])
                }
