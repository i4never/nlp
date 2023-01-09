import numpy as np
import tensorflow as tf
from typing import Optional

supported_activation = {
    'gelu',
    'relu',
    'sigmoid',
    'silu',
    'swish',
    None
}


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def silu(x, beta=1.0):
    return x * tf.nn.sigmoid(beta * x)


def get_activation(activation_name: Optional[str]):
    if activation_name not in supported_activation:
        raise ValueError(f"还没有支持激活函数: {activation_name}")
    if activation_name == 'gelu':
        return gelu
    elif activation_name == 'relu':
        return tf.nn.relu
    elif activation_name == 'sigmoid':
        return tf.nn.sigmoid
    elif activation_name in ['silu', 'swish']:
        return silu
    elif activation_name is None:
        return None
    raise ValueError(f"get_activation错误 {activation_name}")
