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

def roi_pool_cpu(frame, bbox, pool_size):
    frame_h, frame_w = frame.shape[:2]

    x1 = int(bbox[0] * frame_w)
    y1 = int(bbox[1] * frame_h)
    w1 = int(bbox[2] * frame_w)
    h1 = int(bbox[3] * frame_h)

    if (w1 <= 0):
        w1 = 1
    if (h1 <= 0):
        h1 = 1

    slc = frame[y1:y1+h1,x1:x1+w1,...]

    if len(slc.shape) == 3:
        slc2 = np.empty((pool_size, pool_size, slc.shape[2]), dtype=slc.dtype)
        for i in range(slc.shape[2]):
            slc2[:, :, i] = imresize(slc[:, :, i], (pool_size, pool_size), 'bilinear')
        slc = slc2
    else:
        slc = imresize(slc, (pool_size, pool_size), 'bilinear')

    return slc

def process_coco(coco, img_path):
    res = []
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)
    processed = 0
    iter1 = 0

    fake_msk = np.zeros((14, 14), dtype=np.uint8).astype('float32')

    imgs = imgs[:1000]

    for img in imgs:
        iter1 += 1
        processed += 1
        if iter1 > 1000:
            iter1 = 0
            print("processed", processed, len(imgs))

        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        frame_w = img['width']
        frame_h = img['height']
        rois = []
        msks = []
        for ann in anns:
            if ('bbox' in ann) and (ann['bbox'] != []) and ('segmentation' in ann):
                bbox = [int(xx) for xx in ann['bbox']]
                bbox[0] /= frame_w
                bbox[1] /= frame_h
                bbox[2] /= frame_w
                bbox[3] /= frame_h

                m = coco.annToMask(ann)

                if m.max() < 1:
                    continue

                if ann['iscrowd']:
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != frame_h or m.shape[1] != frame_w:
                        m = np.ones([frame_h, frame_w], dtype=bool)

                msk = roi_pool_cpu(m, bbox, 14)

                if (np.count_nonzero(msk) == 0):
                    continue

                assert(len(rois) < my_num_rois)

                x1 = np.float32(bbox[0])
                y1 = np.float32(bbox[1])
                w1 = np.float32(bbox[2])
                h1 = np.float32(bbox[3])

                rois.append([y1, x1, y1 + h1, x1 + w1])
                msks.append(msk.astype('float32'))
        for _ in range(my_num_rois - len(rois)):
            rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
            msks.append(fake_msk)
        msks = np.array(msks)
        msks = msks[..., np.newaxis]
        res.append((img['file_name'], img_path, np.array(rois), msks))

    return res

def fit_generator(coco, imgs, batch_size):
    while True:
        shuffle(imgs)
        for k in range(len(imgs) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(imgs):
                j = - j % len(imgs)
            batch = imgs[i:j]
            x1 = []
            x2 = []
            y = []
            for img_name, _, rois, msks in batch:
                conv = np.load("masknet_data/" + img_name.replace('.jpg', '.npz'))['arr_0']
                x1.append(conv)
                x2.append(rois)
                y.append(msks)
            gc.collect()
            yield ([np.array(x1), np.array(x2)], np.array(y))

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

if __name__ == "__main__":
    img_input = Input(shape=(my_inp_size, my_inp_size, 1024))
    roi_input = Input(shape=(my_num_rois, 4))

    roi_pool_layer = RoiPoolingConv(7, my_num_rois)([img_input, roi_input])

    x = TimeDistributed(Conv2D(2048, (3, 3), activation='relu', padding='same'))(roi_pool_layer)
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), activation='relu', strides=2))(x)
    x = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=1))(x)

    model = Model(inputs=[img_input, roi_input], outputs=x)
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(loss=[my_loss], optimizer=optimizer, metrics=['accuracy'])

    bdir = '../darknet/scripts/coco'
    train_coco = COCO(bdir + "/annotations/person_keypoints_train2014.json")
    val_coco = COCO(bdir + "/annotations/person_keypoints_val2014.json")
    train_imgs = process_coco(train_coco, bdir + "/images/train2014")
    val_imgs = process_coco(val_coco, bdir + "/images/val2014")

    batch_size = 64

    train_data = fit_generator(train_coco, train_imgs, batch_size)

    validation_data = fit_generator(val_coco, val_imgs[:100], batch_size)

    callbacks = []
    callbacks.append(ModelCheckpoint(filepath="weights.hdf5", monitor='val_loss', save_best_only=True))

    model.fit_generator(train_data,
        steps_per_epoch=len(train_imgs) / batch_size,
        validation_steps=len(val_imgs) / batch_size,
        epochs=2,
        validation_data=validation_data,
        use_multiprocessing=False,
        verbose=1,
        callbacks=callbacks)

    print("Done!")
