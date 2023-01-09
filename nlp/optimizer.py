from typing import Any, Optional

import re

import tensorflow as tf

from nlp.common.registrable import Registrable
from pydantic import BaseModel, validator


def create_learning_rate(init_lr, num_train_steps=None, num_warmup_steps=None):
    global_step = tf.train.get_or_create_global_step()
    tf.identity(global_step, "step")

    # Implements linear decay of the learning rate.
    if num_train_steps:
        learning_rate = tf.train.polynomial_decay(
            # init_lr,
            init_lr * num_train_steps / (num_train_steps - num_warmup_steps),
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
    else:
        learning_rate = init_lr

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    tf.summary.scalar('learning_rate', learning_rate)

    return tf.identity(learning_rate, "lr")


class Optimizer(BaseModel, tf.train.Optimizer, Registrable):
    trainable_var_selector: str = ".*"
    step_n: int = 10000

    def get_vars(self, vars):
        return [var for var in vars if re.match(self.trainable_var_selector, var.name)]

    @classmethod
    def average_gradients(cls, dis_grads):
        if len(dis_grads) == 1:
            return dis_grads[0]
        average_grads = []
        for grads in zip(*dis_grads):
            if grads == (None,) * len(grads):
                average_grads.append(None)
            else:
                new_grads = []
                for g in grads:
                    expanded_g = tf.expand_dims(g, 0)
                    new_grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=new_grads)
                grad = tf.reduce_mean(grad, 0)

                average_grads.append(grad)
        return average_grads

    @classmethod
    def from_params(cls, **kwargs):
        if 'name' not in kwargs:
            raise ValueError(f"初始化Optimizer需要提供name")
        return cls.by_name(kwargs['name'])(**kwargs)


@Optimizer.register("adam_weight_decay_optimizer", exist_ok=True)
class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""
    init_lr: float = 1e-4
    warmup_step_n: int = 1000
    weight_decay_rate: float = 0.
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-6
    exclude_from_weight_decay: Optional[str] = None
    learning_rate: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def get_learning_rate(self):
        if self.learning_rate is None:
            self.learning_rate = create_learning_rate(self.init_lr, self.step_n, self.warmup_step_n)
        return self.learning_rate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.get_learning_rate() * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v),
                 global_step.assign(global_step + 1)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


# class AdamWeightDecayOptimizer(tf.train.Optimizer):
#     """A basic Adam optimizer that includes "correct" L2 weight decay."""
#
#     def __init__(self,
#                  learning_rate,
#                  weight_decay_rate=0.0,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-6,
#                  exclude_from_weight_decay=None,
#                  name="AdamWeightDecayOptimizer"):
#         """Constructs a AdamWeightDecayOptimizer."""
#         super(AdamWeightDecayOptimizer, self).__init__(False, name)
#
#         self.learning_rate = learning_rate
#         self.weight_decay_rate = weight_decay_rate
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.epsilon = epsilon
#         self.exclude_from_weight_decay = exclude_from_weight_decay
#
#     def apply_gradients(self, grads_and_vars, global_step=None, name=None):
#         """See base class."""
#         assignments = []
#         for (grad, param) in grads_and_vars:
#             if grad is None or param is None:
#                 continue
#
#             param_name = self._get_variable_name(param.name)
#
#             m = tf.get_variable(
#                 name=param_name + "/adam_m",
#                 shape=param.shape.as_list(),
#                 dtype=tf.float32,
#                 trainable=False,
#                 initializer=tf.zeros_initializer())
#             v = tf.get_variable(
#                 name=param_name + "/adam_v",
#                 shape=param.shape.as_list(),
#                 dtype=tf.float32,
#                 trainable=False,
#                 initializer=tf.zeros_initializer())
#
#             # Standard Adam update.
#             next_m = (
#                     tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
#             next_v = (
#                     tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
#                                                               tf.square(grad)))
#
#             update = next_m / (tf.sqrt(next_v) + self.epsilon)
#
#             # Just adding the square of the weights to the loss function is *not*
#             # the correct way of using L2 regularization/weight decay with Adam,
#             # since that will interact with the m and v parameters in strange ways.
#             #
#             # Instead we want ot decay the weights in a manner that doesn't interact
#             # with the m/v parameters. This is equivalent to adding the square
#             # of the weights to the loss with plain (non-momentum) SGD.
#             if self._do_use_weight_decay(param_name):
#                 update += self.weight_decay_rate * param
#
#             update_with_lr = self.learning_rate * update
#
#             next_param = param - update_with_lr
#
#             assignments.extend(
#                 [param.assign(next_param),
#                  m.assign(next_m),
#                  v.assign(next_v),
#                  global_step.assign(global_step + 1)])
#         return tf.group(*assignments, name=name)
#
#     def _do_use_weight_decay(self, param_name):
#         """Whether to use L2 weight decay for `param_name`."""
#         if not self.weight_decay_rate:
#             return False
#         if self.exclude_from_weight_decay:
#             for r in self.exclude_from_weight_decay:
#                 if re.search(r, param_name) is not None:
#                     return False
#         return True
#
#     def _get_variable_name(self, param_name):
#         """Get the variable name from the tensor name."""
#         m = re.match("^(.*):\\d+$", param_name)
#         if m is not None:
#             param_name = m.group(1)
#         return param_name


def average_gradients(dis_grads):
    if len(dis_grads) == 1:
        return dis_grads[0]
    average_grads = []
    for grads in zip(*dis_grads):
        if grads == (None,) * len(grads):
            average_grads.append(None)
        else:
            new_grads = []
            for g in grads:
                expanded_g = tf.expand_dims(g, 0)
                new_grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=new_grads)
            grad = tf.reduce_mean(grad, 0)

            average_grads.append(grad)
    return average_grads


def create_optimizer(init_lr, num_train_steps=None, num_warmup_steps=None):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()
    tf.identity(global_step, "step")

    # Implements linear decay of the learning rate.
    if num_train_steps:
        learning_rate = tf.train.polynomial_decay(
            init_lr,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
    else:
        learning_rate = init_lr

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    tf.summary.scalar('learning_rate', learning_rate)

    tf.identity(learning_rate, "lr")
    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    # optimizer = AdamWeightDecayOptimizer(
    #     learning_rate=learning_rate,
    #     weight_decay_rate=0.01,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-6,
    #     exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    optimizer = AdamWeightDecayOptimizer(learning_rate)

    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    return optimizer


if __name__ == '__main__':
    optimizer = Optimizer.by_name("adam_weight_decay_optimizer")()
    print(optimizer)
