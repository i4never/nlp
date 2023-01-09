import tensorflow as tf
from pydantic import validator
from typing import Optional, Dict

from nlp.layer.classification import classification_outputs
from nlp.layer.dropout import dropout
from nlp.model.model import Model


@Model.register("multi_class_classifier", exist_ok=True)
class MultiClassClassifier(Model):
    label_n: int
    encoder: Model

    class Config:
        arbitrary_types_allowed = True

    @validator("encoder", pre=True)
    def load_encoder(cls, encoder_params):
        return Model.from_params(**encoder_params)

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        from nlp.model.bert import Bert
        if type(self.encoder) in [Bert]:
            encoder_output = self.encoder.forward(inputs)
            batch_encoder_out = encoder_output['batch_encoder_out']
        else:
            raise ValueError(f"{self.__class__} 没有适配encoder: {self.encoder.__class__}")

        with tf.variable_scope("sequence/cls", reuse=tf.AUTO_REUSE):
            batch_logit = classification_outputs(batch_encoder_out, self.label_n,
                                                 get_new_initializer_foo=self.encoder.get_new_initializer)
            batch_predicted = tf.argmax(batch_logit, axis=-1)
            batch_logit = tf.identity(batch_logit, "batch_logit")
            batch_predicted = tf.identity(batch_predicted, "batch_predicted")
        return {"batch_logit": batch_logit, "batch_predicted": batch_predicted}

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        batch_label_ids = tf.identity(labels["batch_label_ids"], name="batch_label_ids")
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            batch_loss = tf.losses.sparse_softmax_cross_entropy(batch_label_ids, outputs["batch_logit"])
        return batch_loss

    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        return {
            "acc": tf.metrics.accuracy(predictions=outputs["batch_predicted"],
                                       labels=labels["batch_label_ids"]),
            "recall": tf.metrics.recall(predictions=outputs["batch_predicted"],
                                        labels=labels["batch_label_ids"])
        }


if __name__ == '__main__':
    import numpy as np
    from nlp.utils import init_tvars_from_checkpoint
    from nlp.common.registrable import import_all_modules_for_register

    import_all_modules_for_register()

    np.random.seed(12345)

    model = Model.from_params(
        **{
            "name": "multi_class_classifier",
            "encoder":
                {
                    "name": "rec_bert_linear",
                    "vocab_size": 18018,
                    "pooler": "first"
                },
            "label_n": 1
        }
    )
    token_phd = tf.placeholder(shape=(None, None), dtype=tf.int64)

    output = model.forward({"batch_token_ids": token_phd})

    init_tvars_from_checkpoint(tf.trainable_variables(),
                               '/Users/i4never/Documents/workspace/model-finetune/saved_model/model.ckpt-960000',
                               'linear_tfm/.*',
                               None)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for _ in range(10):
            np.random.seed(12345)
            ids = np.random.randint(0, 1231, (3, 4096))
            o = sess.run(output, feed_dict={token_phd: ids})
            for k, v in o.items():
                print(k)
                print(v)
