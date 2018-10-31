""" Reference : https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19 """


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import keras.layers as layers

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)