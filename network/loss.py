"""Collection of loss functions."""

import tensorflow as tf
from util import util

class NerfLoss():
    """Copy of the loss setup of NeRF."""

    def __init__(self, loss_fn: str='network.loss.mse') -> None:
        self.loss = util.get_attr_from_path(loss_fn)

    def __call__(self, color_true: tf.Tensor, color_pred: tf.Tensor, color_pred_coarse: tf.Tensor=None, **kwargs) -> tf.Tensor:
        loss = self.loss(color_true, color_pred)

        if color_pred_coarse is not None:
            loss += self.loss(color_true, color_pred_coarse)

        return loss

class AlphaLoss():
    """Copy of the loss setup of NeRF."""

    def __init__(self, loss_fn: str='network.loss.mse', alpha_loss_fn: str=None, gamma: float=1, filter_color_loss: bool=True, use_hard_mask: bool=True) -> None:
        self.loss = util.get_attr_from_path(loss_fn)
        self.alpha_loss = self.loss if alpha_loss_fn is None else util.get_attr_from_path(alpha_loss_fn)
        self.gamma = gamma
        self.filter_color_loss = filter_color_loss
        self.use_hard_mask = use_hard_mask

    def __call__(self, color_true: tf.Tensor, alpha_true: tf.Tensor, color_pred: tf.Tensor, alpha_pred: tf.Tensor, color_pred_coarse: tf.Tensor=None, alpha_pred_coarse: tf.Tensor=None, **kwargs) -> tf.Tensor:
        if self.filter_color_loss:
            if self.use_hard_mask:
                alpha_mask = tf.cast(alpha_true[...,None] > 0, tf.float32)
            else:
                alpha_mask = alpha_true[...,None]
            color_true *= alpha_mask
            color_pred *= alpha_mask

        loss = self.loss(color_true, color_pred) 
        loss += self.gamma * self.alpha_loss(alpha_true, alpha_pred)

        if color_pred_coarse is not None:
            if self.filter_color_loss:
                color_pred_coarse *= alpha_mask

            loss += self.loss(color_true, color_pred_coarse)
            loss += self.gamma * self.alpha_loss(alpha_true, alpha_pred_coarse)

        return loss

def mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean squared error."""

    return tf.math.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def smape(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float=1e-2):
    """Symmetric mean absolute percentage error."""

    return tf.math.reduce_mean(tf.abs(y_true - y_pred) / (y_true + y_pred + eps))