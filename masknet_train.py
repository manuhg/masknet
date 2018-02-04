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
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, SGD, RMSprop
from pycocotools.coco import COCO
from keras.metrics import binary_accuracy
from random import shuffle
import gc, math
import keras.backend as K
from scipy.misc import imresize
from datetime import datetime
import threading

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import masknet

def roi_pool_cpu(frame, bbox, pool_size):
    frame_h, frame_w = frame.shape[:2]

    x1 = int(bbox[0] * frame_w)
    y1 = int(bbox[1] * frame_h)
    w1 = int(bbox[2] * frame_w)
    h1 = int(bbox[3] * frame_h)

    #if (w1 <= 0):
    #    w1 = 1
    #if (h1 <= 0):
    #    h1 = 1

    slc = frame[y1:y1+h1,x1:x1+w1,...]

    if (w1 <= 0) or (h1 <= 0):
        assert(np.count_nonzero(slc) == 0)
        return slc

    slc = imresize(slc.astype(float), (pool_size, pool_size), 'nearest') / 255.0

    return slc

def process_coco(coco, img_path, limit):
    res = []
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)
    processed = 0
    iter1 = 0

    fake_msk = np.zeros((masknet.my_msk_inp * 2, masknet.my_msk_inp * 2), dtype=np.uint8).astype('float32')

    if limit:
        imgs = imgs[:limit]

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
                    continue
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != frame_h or m.shape[1] != frame_w:
                        m = np.ones([frame_h, frame_w], dtype=bool)

                msk = roi_pool_cpu(m, bbox, masknet.my_msk_inp * 2)

                if (np.count_nonzero(msk) == 0):
                    continue

                assert(len(rois) < masknet.my_num_rois)

                x1 = np.float32(bbox[0])
                y1 = np.float32(bbox[1])
                w1 = np.float32(bbox[2])
                h1 = np.float32(bbox[3])

                rois.append([y1, x1, y1 + h1, x1 + w1])
                msks.append(msk)
        if (len(rois) > 0):
            for _ in range(masknet.my_num_rois - len(rois)):
                rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
                msks.append(fake_msk)
            msks = np.array(msks)
            msks = msks[..., np.newaxis]
            res.append((img['file_name'], img_path, np.array(rois), msks))

    return res

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def fit_generator(imgs, batch_size):
    ii = 0
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
                conv = np.load("masknet_data_28/" + img_name.replace('.jpg', '.npz'))['arr_0']
                x1.append(conv)
                x2.append(rois)
                y.append(msks)
            #gc.collect()
            #print("yield",ii)
            ii += 1
            yield ([np.array(x1), np.array(x2)], np.array(y))

def my_accuracy(y_true, y_pred):
    mask_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    mask_shape = tf.shape(y_true)
    y_true = K.reshape(y_true, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))

    sm = tf.reduce_sum(y_true, [1,2,3])

    ix = tf.where(sm > 0)[:, 0]

    y_true = tf.gather(y_true, ix)
    y_pred = tf.gather(y_pred, ix)

    return binary_accuracy(y_true, y_pred)

if __name__ == "__main__":
    model = masknet.create_model()
    model.summary()
    model.compile(loss=[masknet.my_loss], optimizer='adam', metrics=[my_accuracy])
    #model.load_weights("weights50_1.hdf5")

    bdir = '../darknet/scripts/coco'
    train_coco = COCO(bdir + "/annotations/person_keypoints_train2014.json")
    val_coco = COCO(bdir + "/annotations/person_keypoints_val2014.json")
    train_imgs = process_coco(train_coco, bdir + "/images/train2014", 10000)
    val_imgs = process_coco(val_coco, bdir + "/images/val2014", 1500)

    #train_imgs += val_imgs[5000:]
    #val_imgs = val_imgs[:5000]

    batch_size = 16

    train_data = fit_generator(train_imgs, batch_size)

    validation_data = fit_generator(val_imgs, batch_size)

    #lr_schedule = lambda epoch: 0.001 if epoch < 120 else 0.0001
    lr_schedule = lambda epoch: 0.0001
    #lr_schedule = lambda epoch: 1e-5
    callbacks = [LearningRateScheduler(lr_schedule)]
    callbacks.append(ModelCheckpoint(filepath="weights.hdf5", monitor='val_loss', save_best_only=True))

    model.fit_generator(train_data,
        steps_per_epoch=len(train_imgs) / batch_size,
        validation_steps=len(val_imgs) / batch_size,
        epochs=160,
        validation_data=validation_data,
        max_queue_size=50,
        workers=1, use_multiprocessing=False,
        verbose=1,
        callbacks=callbacks)

    print("Done!")
