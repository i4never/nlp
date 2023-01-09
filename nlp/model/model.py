import enum

import logging
import os
from typing import List, Optional, Dict, Any, Tuple

import tensorflow as tf
from pydantic import BaseModel

from nlp.common.registrable import Registrable
from nlp.optimizer import Optimizer
from nlp.utils import init_tvars_from_checkpoint

logger = logging.getLogger(__name__)


class Model(Registrable, BaseModel):
    is_training: bool = False
    dropout_rate: float = 0.2

    @classmethod
    def split_input(cls, features, n: int):
        """
        Split input for multi gpu train
        :param features:
        :param n:
        :return:
        """
        features_split = [dict() for _ in range(n)]
        for p in ['inputs', 'outputs', 'weights']:
            if p not in features:
                continue
            for k, v in features[p].items():
                v_split = tf.split(v, n)
                for i in range(n):
                    features_split[i][p] = features_split[i].get(p, dict())
                    features_split[i][p][k] = v_split[i]
        return features_split

    @classmethod
    def merge_output(cls, outputs):
        """
        Merge output for multi gpu eval
        确保返回的第一个纬度为batch_size
        :param outputs:
        :return:
        """
        outputs_merged = dict()
        for output in outputs:
            for k, v in output.items():
                outputs_merged[k] = outputs_merged.get(k, list()) + [output[k]]
        return {k: tf.concat(v, axis=0) for k, v in outputs_merged.items()}

    def build_model_fn(self, run_args: Dict, optimizers: List[Optimizer], evaluation_hooks: Optional = None):
        from nlp.hooks.hooks import LossLogHooks
        def model_fn(features, labels, mode):
            assert "inputs" in features
            # Predict mode for output
            if mode == tf.estimator.ModeKeys.PREDICT:
                outputs = self.forward(features['inputs'])
                logger.info(f"{self.__class__.__name__} predict output: {outputs}")
                return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=outputs)

            gpu_n = len(run_args['gpus'].split(','))
            init_ckpt = run_args.get('init_ckpt')
            if init_ckpt is not None:
                assert os.path.exists('/'.join(init_ckpt.split('/')[:-1]))

            features_split = self.split_input(features, gpu_n)

            grads_in_each_gpu = list()
            loss_in_each_gpu = list()

            outputs_split = list()
            for gpu_idx in range(gpu_n):
                with tf.device(f"/gpu:{gpu_idx}"):
                    outputs = self.forward(features_split[gpu_idx]['inputs'])
                    loss = self.loss(outputs,
                                     features_split[gpu_idx]['outputs'],
                                     features_split[gpu_idx].get('weights'))
                    loss_in_each_gpu.append(loss)
                    if mode == tf.estimator.ModeKeys.EVAL:
                        outputs_split.append(outputs)
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        trainable_vars = tf.trainable_variables()
                        optimizer_grads = [
                            [g for g, v in optimizer.compute_gradients(loss, optimizer.get_vars(trainable_vars))]
                            for optimizer in optimizers]
                        grads_in_each_gpu.append(optimizer_grads)

            loss = tf.reduce_mean(tf.stack(loss_in_each_gpu)) if len(loss_in_each_gpu) > 1 else loss_in_each_gpu[0]
            loss = tf.identity(loss, f"{mode}_loss")

            if mode == tf.estimator.ModeKeys.TRAIN:
                if init_ckpt:
                    logger.info(f"从{init_ckpt}加载")
                    init_tvars_from_checkpoint(trainable_vars, init_ckpt, run_args["init_vars_re"])
                train_ops = list()
                global_step = tf.train.get_or_create_global_step()
                for opt_idx, optimizer in enumerate(optimizers):
                    grads = [grads[opt_idx] for grads in grads_in_each_gpu]
                    grads_avg = optimizer.average_gradients(grads)
                    grads_avg, _ = tf.clip_by_global_norm(grads_avg, clip_norm=1.)
                    train_op = optimizer.apply_gradients(list(zip(grads_avg, optimizer.get_vars(trainable_vars))),
                                                         global_step)
                    train_ops.append(train_op)
                return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                                  loss=loss,
                                                  train_op=tf.group(train_ops),
                                                  training_hooks=[
                                                      LossLogHooks(
                                                          log_step_count_steps=run_args["log_step_count_steps"],
                                                          loss_op_name=f"{mode}_loss:0",
                                                          is_train=True)
                                                  ])

            if mode == tf.estimator.ModeKeys.EVAL:
                outputs = self.merge_output(outputs_split)
                metric_ops = self.metric(outputs, features['outputs'], features.get('weights'))
                return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                                  loss=loss,
                                                  eval_metric_ops=metric_ops,
                                                  evaluation_hooks=[LossLogHooks(log_step_count_steps=1,
                                                                                 loss_op_name=f"{mode}_loss:0",
                                                                                 is_train=False)] + (
                                                                               evaluation_hooks or list()))

        return model_fn

    @property
    def dropout_prob(self):
        return self.dropout_rate if self.is_training else 0.

    def forward(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        raise NotImplementedError

    def loss(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
             weights: Optional[Dict[str, tf.Tensor]] = None):
        raise NotImplementedError

    def metric(self, outputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
               weights: Optional[Dict[str, tf.Tensor]] = None):
        raise NotImplementedError

    @classmethod
    def from_params(cls, **kwargs):
        if 'name' not in kwargs:
            raise ValueError(f"初始化模型需要提供name")
        return cls.by_name(kwargs['name'])(**kwargs)
