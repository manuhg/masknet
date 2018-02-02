#!/usr/bin/python3

import cv2
import numpy as np
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from pycocotools.coco import COCO
from random import shuffle
import gc, math
import keras.backend as K
from scipy.misc import imresize
from datetime import datetime

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf

my_num_rois = 32
my_inp_size = 19

class RoiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert(self.dim_ordering == 'tf')

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        rois_flattened = K.reshape(rois, (-1, 4))

        shape = tf.shape(rois_flattened)
        box_indices = tf.range(0, shape[0]) // self.num_rois

        res = tf.image.crop_and_resize(
            img, rois_flattened, box_indices, (self.pool_size, self.pool_size),
            method="bilinear")

        res = K.reshape(res, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return res

def my_loss(y_true, y_pred):
    mask_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    mask_shape = tf.shape(y_true)
    y_true = K.reshape(y_true, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))

    sm = tf.reduce_sum(y_true, 0)

    ix = tf.where(sm > 0)[:, 0]

    y_true = tf.gather(y_true, ix)
    y_pred = tf.gather(y_pred, ix)

    loss = K.binary_crossentropy(target=y_true, output=y_pred)
    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss

def create_model():
    img_input = Input(shape=(my_inp_size, my_inp_size, 1024))
    roi_input = Input(shape=(my_num_rois, 4))

    roi_pool_layer = RoiPoolingConv(7, my_num_rois)([img_input, roi_input])

    x = TimeDistributed(Conv2D(2048, (3, 3), activation='relu', padding='same'))(roi_pool_layer)
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), activation='relu', strides=2))(x)
    x = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=1))(x)

    return Model(inputs=[img_input, roi_input], outputs=x)
