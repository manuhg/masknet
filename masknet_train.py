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
import masknet

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

                assert(len(rois) < masknet.my_num_rois)

                x1 = np.float32(bbox[0])
                y1 = np.float32(bbox[1])
                w1 = np.float32(bbox[2])
                h1 = np.float32(bbox[3])

                rois.append([y1, x1, y1 + h1, x1 + w1])
                msks.append(msk.astype('float32'))
        for _ in range(masknet.my_num_rois - len(rois)):
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

if __name__ == "__main__":
    model = masknet.create_model()
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(loss=[masknet.my_loss], optimizer=optimizer, metrics=['accuracy'])

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
