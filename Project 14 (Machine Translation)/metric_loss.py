import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric

class MaskedCELoss(Loss):
  def __init__(self, from_logits: bool, reduction: Optional[str]):
    super(MaskedCELoss, self).__init__()

    assert reduction in ["none", "sum_over_batch_size", "sum"], "Reduction must be either none, sum_over_batch_size, or sum."

    self.reduction=reduction
    self.loss_object = SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')

  def call(self, y_true, y_pred):

    mask = y_true != 0
    loss = self.loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    if self.reduction == "sum_over_batch_size":
      total_loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
      return total_loss

    elif self.reduction == "sum":
      total_loss = tf.reduce_sum(loss)
      return total_loss

    else:
      return loss


class MaskedAccuracy(Metric):
  def __init__(self, from_logits=bool):
    super(MaskedAccuracy, self).__init__()

    self.from_logits = from_logits
    self.true_positive = self.add_variable(shape=(), initializer='zeros', name='true_positives')
    self.mask_counts = self.add_variable(shape=(), initializer='zeros', name='mask_counter')

  def update_state(self, y_true, y_pred, sample_weight=None):

    if self.from_logits:
      y_pred = tf.nn.softmax(y_pred, axis=-1)

    y_pred = tf.argmax(y_pred, axis=2)
    y_true = tf.cast(y_true, y_pred.dtype)

    corrects = y_true == y_pred

    mask = y_true != 0

    corrects = corrects & mask

    corrects = tf.cast(corrects, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    self.true_positive.assign_add(tf.reduce_sum(corrects))
    self.mask_counts.assign_add(tf.reduce_sum(mask))

  def result(self):
    return self.true_positive/self.mask_counts

  def reset_states(self):
    self.true_positive.assign(0.)
    self.mask_counts.assign(0.)
