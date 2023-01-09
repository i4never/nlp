"""
TODO: 这里的EvalHook都没有很好地抽象接口&工程化
"""
from typing import List

import numpy as np
import tensorflow as tf


class BIOEvalHook(tf.train.SessionRunHook):
    def __init__(self):
        self.metric = dict()
        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0

    def _reset(self):
        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0

    def batch_bio_seq_to_tuple(self, batch_seq, batch_length):
        batch_entities = list()
        batch_length = np.reshape(batch_length, -1).tolist()
        for seq, length in zip(batch_seq, batch_length):
            entities, starting, pt = [], False, 0
            for idx, label in enumerate(seq):
                if idx >= length:
                    break
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
        return tf.estimator.SessionRunArgs([f"viterbi_seq:0",
                                            f"batch_y:0",
                                            f"batch_token_ids:0"])

    def after_run(self, run_context, run_values):
        batch_predicted, batch_label, batch_length = run_values.results
        batch_predicted_in_tuple = self.batch_bio_seq_to_tuple(batch_predicted, batch_length)
        batch_true_in_tuple = self.batch_bio_seq_to_tuple(batch_label, batch_length)

        # exact match
        self.tp += sum([len(set(predicted) & set(true)) for predicted, true in
                        zip(batch_predicted_in_tuple, batch_true_in_tuple)])
        self.tp_fp += sum([len(predicted) for predicted in batch_true_in_tuple])
        self.tp_fn += sum([len(true) for true in batch_predicted_in_tuple])

        # self.metric['f1_exact_match'] = 2 * tp_exact_match / (tp_fp_exact_match + tp_fn_exact_match + 1e-9)
        # self.metric['precision_exact_match'] = tp_exact_match / (tp_fp_exact_match + 1e-9)
        # self.metric['recall_exact_match'] = tp_exact_match / (tp_fn_exact_match + 1e-9)
        # return self.metric

    def end(self, session):
        self.metric['f1_exact_match'] = 2 * self.tp / (self.tp_fp + self.tp_fn + 1e-9)
        self.metric['precision_exact_match'] = self.tp / (self.tp_fp + 1e-9)
        self.metric['recall_exact_match'] = self.tp / (self.tp_fn + 1e-9)

        # summary_writer = tf.summary.FileWriter(self.ckpt_path, session.graph)
        # summary_writer.add_summary(self.metric['f1_exact_match'], global_step=self._step)

        print(self.metric)

        self._reset()


class SpanEvalHook(tf.train.SessionRunHook):
    def __init__(self, predicted_op, true_op, thd=0.):
        self.metric = dict()
        self.predicted_op = predicted_op
        self.true_op = true_op
        self.thd = thd

        self.span_tp = 0
        self.span_tp_fp = 0
        self.span_tp_fn = 0

    def _reset(self):
        self.span_tp = 0
        self.span_tp_fp = 0
        self.span_tp_fn = 0
        self.metric = dict()

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs({"predicted": self.predicted_op,
                                            "label": self.true_op})

    def after_run(self, run_context, run_values, batch_predicted_in_tuple=None):
        outputs = run_values.results
        predicted = outputs['predicted']
        label = outputs['label']

        predicted[predicted >= self.thd] = 1
        predicted[predicted < self.thd] = 0

        for pred, true in zip(predicted, label):
            pred_spans = set()
            true_spans = set()
            for i in range(0, pred.shape[1]):
                pred_spans |= {tuple(list(span) + [i // 2]) for span in self.decode(pred[:, i, 0], pred[:, i, 1])}
                true_spans |= {tuple(list(span) + [i // 2]) for span in self.decode(true[:, i], true[:, i + 1])}

            self.span_tp += len(pred_spans & true_spans)
            self.span_tp_fp += len(pred_spans)
            self.span_tp_fn += len(true_spans)

    @staticmethod
    def decode(start_matrix, end_matrix):
        points = [(i, 'B') for i in range(start_matrix.shape[0]) if start_matrix[i] != 0]
        points += [(i, 'E') for i in range(end_matrix.shape[0]) if end_matrix[i] != 0]
        points = sorted(points)

        spans = list()
        for i in range(len(points) - 1):
            if points[i][-1] == 'B' and points[i + 1][-1] == 'E':
                spans.append((points[i][0], points[i + 1][0]))
        return spans

    def end(self, session):
        self.metric['span_precision'] = self.span_tp / (self.span_tp_fp + 1e-9)
        self.metric['span_recall'] = self.span_tp / (self.span_tp_fn + 1e-9)
        self.metric['span_f1'] = 2 * self.span_tp / (self.span_tp_fp + self.span_tp_fn + 1e-9)

        for k, v in self.metric.items():
            print(k, f"{v * 100:.2f}%")
        self._reset()


class Span2DEvalHook(tf.train.SessionRunHook):
    def __init__(self, predicted_op, true_op, mask_op, thd=0.):
        self.metric = {"pointer_precision": 0., "pointer_recall": 0., "pointer_f1": 0.,
                       "span_precision": 0., "span_recall": 0., "span_f1": 0.}
        self.predicted_op = predicted_op
        self.true_op = true_op
        self.mask_op = mask_op
        self.thd = thd

        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0

        self.span_tp = 0
        self.span_tp_fp = 0
        self.span_tp_fn = 0

    def _reset(self):
        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0
        self.span_tp = 0
        self.span_tp_fp = 0
        self.span_tp_fn = 0
        self.metric = {"pointer_precision": 0., "pointer_recall": 0., "pointer_f1": 0.,
                       "span_precision": 0., "span_recall": 0., "span_f1": 0.}

    # def after_create_session(self, session, coord):
    #     for k, v in self.metric.items():
    #         tf.summary.scalar(k, v)
    #     self._reset()

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs({"predicted": self.predicted_op,
                                            "label": self.true_op,
                                            "mask": self.mask_op})

    def after_run(self, run_context, run_values):
        outputs = run_values.results
        predicted = outputs['predicted']
        predicted[predicted >= self.thd] = 1.
        predicted[predicted < self.thd] = 0.
        label = outputs['label']
        mask = outputs['mask']

        predicted = predicted * np.expand_dims(mask, axis=-1)

        self.tp += (predicted * label).sum()
        self.tp_fp += predicted.sum()
        self.tp_fn += label.sum()

        for pred, golden in zip(predicted, label):
            pred_spans = set()
            true_spans = set()
            for i in range(0, pred.shape[2], 2):
                pred_spans |= {tuple(list(span) + [i]) for span in self.decode(pred[:, :, i], pred[:, :, i + 1])}
                true_spans |= {tuple(list(span) + [i]) for span in self.decode(golden[:, :, i], golden[:, :, i + 1])}

            # print('true_span', true_spans)
            # print('pred_span', pred_spans)

            self.span_tp += len(pred_spans & true_spans)
            self.span_tp_fp += len(pred_spans)
            self.span_tp_fn += len(true_spans)

    @staticmethod
    def decode(start_matrix, end_matrix):
        points = [(r, c, 'B') for r in range(start_matrix.shape[0]) for c in range(start_matrix.shape[1]) if
                  start_matrix[r, c] != 0]
        points += [(r, c, 'E') for r in range(end_matrix.shape[0]) for c in range(end_matrix.shape[1]) if
                   end_matrix[r, c] != 0]
        points = sorted(points)

        spans = list()
        for i in range(len(points) - 1):
            if points[i][-1] == 'B' and points[i + 1][-1] == 'E':
                spans.append(((points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1])))
        return spans

    def end(self, session):
        print(f"tp: {self.tp} tp_fp: {self.tp_fp} tp_fn: {self.tp_fn}")
        self.metric['pointer_precision'] = self.tp / (self.tp_fp + 1e-9)
        self.metric['pointer_recall'] = self.tp / (self.tp_fn + 1e-9)
        self.metric['pointer_f1'] = 2 * self.tp / (self.tp_fp + self.tp_fn + 1e-9)

        self.metric['span_precision'] = self.span_tp / (self.span_tp_fp + 1e-9)
        self.metric['span_recall'] = self.span_tp / (self.span_tp_fn + 1e-9)
        self.metric['span_f1'] = 2 * self.span_tp / (self.span_tp_fp + self.span_tp_fn + 1e-9)

        for k, v in self.metric.items():
            print(k, f"{v * 100:.2f}%")
        self._reset()


class GlobalPointerEvalHook(tf.train.SessionRunHook):
    def __init__(self, predicted_op, true_op, thd=0.):
        self.metric = dict()
        self.predicted_op = predicted_op
        self.true_op = true_op
        self.thd = thd

        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0

    def _reset(self):
        self.tp = 0
        self.tp_fp = 0
        self.tp_fn = 0
        self.metric = dict()

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs({"predicted": self.predicted_op,
                                            "label": self.true_op})

    def after_run(self, run_context, run_values, batch_predicted_in_tuple=None):
        outputs = run_values.results
        predicted = outputs['predicted']
        label = outputs['label']

        predicted[predicted >= self.thd] = 1
        predicted[predicted < self.thd] = 0

        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        l = predicted.shape[-1]

        mask = np.arange(l)
        mask = mask[:, None] <= mask[None, :]

        predicted = predicted * mask
        label = label * mask

        self.tp += np.sum(predicted * label)
        self.tp_fp += np.sum(predicted)
        self.tp_fn += np.sum(label)

        # print(self.tp, self.tp_fp, self.tp_fn)

    def end(self, session):
        self.metric['precision'] = self.tp / (self.tp_fp + 1e-9)
        self.metric['recall'] = self.tp / (self.tp_fn + 1e-9)
        self.metric['f1'] = 2 * self.tp / (self.tp_fp + self.tp_fn + 1e-9)

        for k, v in self.metric.items():
            print(k, f"{v * 100:.2f}%")
        self._reset()
