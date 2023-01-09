import uuid

from typing import List, Optional, Dict, Any

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, validator
import mlflow
from nlp.common.registrable import Registrable


class Hook(BaseModel, Registrable, tf.train.SessionRunHook):
    @classmethod
    def from_params(cls, **kwargs):
        if 'name' not in kwargs:
            raise ValueError(f"初始化Hook需要提供name")
        return cls.by_name(kwargs['name'])(**kwargs)


@Hook.register("loss_log_hooks", exist_ok=True)
class LossLogHooks(Hook):
    log_step_count_steps: int
    is_train: bool = True
    loss_op_name: str = "loss:0"
    total_losses: List[float] = list()
    step: int = -1
    id: Optional[str] = None

    @validator("id", pre=True)
    def generate_hash(cls, id):
        return str(uuid.uuid1())

    def __hash__(self):
        return self.id.__hash__()

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs([tf.train.get_global_step(),
                                            self.loss_op_name])

    def after_run(self, run_context, run_values):
        step, loss = run_values.results
        self.step = step
        if self.is_train:
            self.total_losses.append(loss)

            if len(self.total_losses) > 0 and len(self.total_losses) % self.log_step_count_steps == 0:
                loss = sum(self.total_losses) / len(self.total_losses)
                self.total_losses.clear()
                mlflow.log_metric(f"train_loss", loss, int(step))
        else:
            self.total_losses.append(loss)

    def end(self, session):
        if not self.is_train:
            loss = sum(self.total_losses) / len(self.total_losses)
            mlflow.log_metric(f"eval_loss", loss, int(self.step))


@Hook.register("multiclass_metric_hooks", exist_ok=True)
class MultiClassMetricHooks(Hook):
    metric: Dict = dict()
    step: int = -1
    batch_golden_op: str = "batch_label_ids:0"
    batch_predicted_op: str = "sequence/cls/batch_predicted:0"
    confusion_matrix: Optional[Any] = None
    label_n: int
    id: Optional[str] = None

    @validator("id", pre=True)
    def generate_hash(cls, id):
        return str(uuid.uuid1())

    def __hash__(self):
        return self.id.__hash__()

    def _reset(self):
        self.confusion_matrix = np.zeros((self.label_n, self.label_n))

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs([tf.train.get_global_step(),
                                            self.batch_golden_op,
                                            self.batch_predicted_op])

    def after_run(self, run_context, run_values):
        if self.confusion_matrix is None:
            self._reset()
        self.step, batch_label, batch_predicted = run_values.results
        batch_label = np.reshape(batch_label, [-1])
        batch_predicted = np.reshape(batch_predicted, [-1])
        for p, t in zip(batch_predicted, batch_label):
            self.confusion_matrix[p, t] += 1

    def end(self, session):
        print(self.confusion_matrix)
        for i in range(self.label_n):
            recall = self.confusion_matrix[i][i] / (np.sum(self.confusion_matrix[:, i]) + 1e-9)
            acc = self.confusion_matrix[i][i] / (np.sum(self.confusion_matrix[i, :]) + 1e-9)
            mlflow.log_metric(f'class_{i}_recall', recall, int(self.step))
            mlflow.log_metric(f'class_{i}_acc', acc, int(self.step))
            mlflow.log_metric(f'class_{i}_f1', 2 * acc * recall / (acc + recall + 1e-9), int(self.step))
        self._reset()


@Hook.register("bio_metric_hooks", exist_ok=True)
class BIOMetricHooks(Hook):
    metric: Dict = dict()
    span_tp: int = 0
    span_tp_fp: int = 0
    span_tp_fn: int = 0
    token_tp: int = 0
    token_tp_fp: int = 0
    token_tp_fn: int = 0
    step: int = -1
    batch_viterbi_decoded_labels_op: str = "crf/batch_viterbi_decoded_labels:0"
    batch_token_labels_predicted_op: str = "crf/batch_token_labels:0"
    batch_token_labels_golden_op: str = "batch_token_labels:0"
    batch_length_mask_op: str = "batch_length_mask:0"
    id: Optional[str] = None

    @validator("id", pre=True)
    def generate_hash(cls, id):
        return str(uuid.uuid1())

    def __hash__(self):
        return self.id.__hash__()

    def _reset(self):
        self.span_tp = 0
        self.span_tp_fp = 0
        self.span_tp_fn = 0

        self.token_tp = 0
        self.token_tp_fp = 0
        self.token_tp_fn = 0

        self.metric = dict()

    def batch_bio_seq_to_tuple(self, batch_seq, batch_length):
        batch_entities = list()
        batch_length = np.reshape(batch_length, -1).tolist()
        for seq, length in zip(batch_seq, batch_length):
            entities, starting, pt = [], False, 0
            for idx, label in enumerate(seq[:length]):
                if label > 0:
                    if label % 2 == 1:
                        starting = True
                        entities.append([label // 2, [idx]])
                    elif starting:
                        entities[-1][-1].append(idx)
                    else:
                        starting = False
                else:
                    starting = False
            batch_entities.append(
                [(entity_idx, min(token_indexes), max(token_indexes) + 1) for entity_idx, token_indexes in entities])
        return batch_entities

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs([tf.train.get_global_step(),
                                            self.batch_viterbi_decoded_labels_op,
                                            self.batch_token_labels_predicted_op,
                                            self.batch_token_labels_golden_op,
                                            self.batch_length_mask_op])

    def after_run(self, run_context, run_values):
        self.step, batch_viterbi_decoded_labels, batch_token_labels_predicted, batch_token_labels_golden, batch_length_mask \
            = run_values.results
        batch_length = np.sum(batch_length_mask, axis=-1)
        batch_predicted_in_tuple = self.batch_bio_seq_to_tuple(batch_viterbi_decoded_labels, batch_length)
        batch_true_in_tuple = self.batch_bio_seq_to_tuple(batch_token_labels_golden, batch_length)

        # exact match
        self.span_tp += sum([len(set(predicted) & set(true)) for predicted, true in
                             zip(batch_predicted_in_tuple, batch_true_in_tuple)])
        self.span_tp_fp += sum([len(predicted) for predicted in batch_predicted_in_tuple])
        self.span_tp_fn += sum([len(true) for true in batch_true_in_tuple])

        # token match
        batch_token_labels_predicted = np.argmax(batch_token_labels_predicted, axis=-1) * batch_length_mask
        batch_token_labels_golden *= batch_length_mask

        self.token_tp += np.sum(
            ((batch_token_labels_predicted > 0) * (batch_token_labels_predicted == batch_token_labels_golden)).astype(
                np.int32))
        self.token_tp_fp += np.sum((batch_token_labels_predicted != 0).astype(np.int32))
        self.token_tp_fn += np.sum((batch_token_labels_golden != 0).astype(np.int32))
        print('after_run', self)

    def end(self, session):
        self.metric['f1_span_match'] = 2 * self.span_tp / (self.span_tp_fp + self.span_tp_fn + 1e-9)
        self.metric['precision_span_match'] = self.span_tp / (self.span_tp_fp + 1e-9)
        self.metric['recall_span_match'] = self.span_tp / (self.span_tp_fn + 1e-9)

        self.metric['f1_token_match'] = 2 * self.token_tp / (self.token_tp_fp + self.token_tp_fn + 1e-9)
        self.metric['precision_token_match'] = self.token_tp / (self.token_tp_fp + 1e-9)
        self.metric['recall_token_match'] = self.token_tp / (self.token_tp_fn + 1e-9)

        # summary_writer = tf.summary.FileWriter(self.ckpt_path, session.graph)
        # summary_writer.add_summary(self.metric['f1_exact_match'], global_step=self._step)
        print(self.metric)
        for k, v in self.metric.items():
            mlflow.log_metric(k, v, int(self.step))
        self._reset()


if __name__ == '__main__':
    loss_log_hooks = Hook.by_name("loss_log_hooks")(log_step_count_steps=100)
    print(loss_log_hooks)
