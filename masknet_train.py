#!/usr/bin/python3

import cv2
import numpy as np
import keras.layers as layers
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
import threading, os, re

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
        bboxs = []
        cocos = []
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
                msks.append(ann)
                bboxs.append(bbox)
                cocos.append(coco)
        if (len(rois) > 0):
            for _ in range(masknet.my_num_rois - len(rois)):
                rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
                msks.append(None)
                bboxs.append(None)
                cocos.append(None)
            res.append((img['file_name'], img_path, np.array(rois), msks, bboxs, cocos))

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

def my_preprocess(im):
    h = 608
    w = 608
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

@threadsafe_generator
def fit_generator(imgs, batch_size):
    ii = 0
    fake_msk = np.zeros((masknet.my_msk_inp * 2, masknet.my_msk_inp * 2), dtype=np.uint8).astype('float32')
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
            for img_name, img_path, rois, anns, bboxs, cocos in batch:
                #x12.append(np.load("masknet_data_17/" + img_name.replace('.jpg', '.npz'))['arr_0'])
                #x13.append(np.load("masknet_data_28/" + img_name.replace('.jpg', '.npz'))['arr_0'])
                #x14.append(np.load("masknet_data_43/" + img_name.replace('.jpg', '.npz'))['arr_0'])
                frame = cv2.imread(img_path + "/" + img_name)
                x1.append(my_preprocess(frame))
                x2.append(rois)

                msks = []
                for k in range(len(bboxs)):
                    if cocos[k] is None:
                        msk = fake_msk
                    else:
                        msk = roi_pool_cpu(cocos[k].annToMask(anns[k]), bboxs[k], masknet.my_msk_inp * 2)
                    msks.append(msk)

                msks = np.array(msks)
                msks = msks[..., np.newaxis]

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

def yolo():
    ALPHA = 0.1
    input_image = layers.Input(shape=(608, 608, 3))
    # Layer 1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                        padding='same', name='conv_1', use_bias=False)(input_image)
    x = layers.BatchNormalization(name='norm_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    # Layer 2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_2')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)


    # Layer 3
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_3')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 4
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_4')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 5
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_5')(x)
    C2 = x= layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_6')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    # Layer 7
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x= layers.BatchNormalization(name='norm_7')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 8
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_8')(x)
    C3 = x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_9')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 10
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_10')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 11
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_11')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    # Layer 12
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_12')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 13
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_13')(x)
    C4 = x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)


    skip_connection = x

    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Layer 14
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_14')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 15
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_15')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 16
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_16')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 17
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_17')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 18
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_18')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 19
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_19')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 20
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_20')(x)
    C5 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    model = Model([input_image], x)

    return model, input_image, C2, C3, C4, C5

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        tmp_model = self.model
        self.model = self.my_model
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.model = tmp_model

if __name__ == "__main__":
    yolo_model, yolo_input, yolo_C2, yolo_C3, yolo_C4, yolo_C5 = yolo()

    weight_reader = WeightReader("bin/yolo.weights")

    weight_reader.reset()
    nb_conv = 20

    for i in range(1, nb_conv+1):
        conv_layer = yolo_model.get_layer('conv_' + str(i))

        if i < nb_conv+1:
            norm_layer = yolo_model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])

        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])
        conv_layer.trainable = False
        norm_layer.trainable = False

    mn_model = masknet.create_model()
    mn_model.summary()

    weight_reader = None

    m_roi_input = Input(shape=(masknet.my_num_rois, 4))

    x = mn_model([yolo_C2, yolo_C3, yolo_C4, yolo_C5, m_roi_input])

    model = Model(inputs=[yolo_input, m_roi_input], outputs=x)
    model.summary()
    model.compile(loss=[masknet.my_loss], optimizer='adam', metrics=[my_accuracy])
    #model.load_weights("weights50_1.hdf5")

    bdir = '../darknet/scripts/coco'
    train_coco = COCO(bdir + "/annotations/person_keypoints_train2014.json")
    val_coco = COCO(bdir + "/annotations/person_keypoints_val2014.json")
    train_imgs = process_coco(train_coco, bdir + "/images/train2014", None)
    val_imgs = process_coco(val_coco, bdir + "/images/val2014", None)

    train_coco = None
    val_coco = None

    train_imgs += val_imgs[5000:]
    val_imgs = val_imgs[:5000]

    batch_size = 16

    train_data = fit_generator(train_imgs, batch_size)

    validation_data = fit_generator(val_imgs, batch_size)

    #lr_schedule = lambda epoch: 0.001 if epoch < 120 else 0.0001
    lr_schedule = lambda epoch: 0.001
    #lr_schedule = lambda epoch: 1e-5
    callbacks = [LearningRateScheduler(lr_schedule)]

    mcp = MyModelCheckpoint(filepath="weights.hdf5", monitor='val_loss', save_best_only=True)
    mcp.my_model = mn_model

    callbacks.append(mcp)

    callbacks.append(ModelCheckpoint(filepath="all_weights.hdf5", monitor='val_loss', save_best_only=True))

    model.fit_generator(train_data,
        steps_per_epoch=len(train_imgs) / batch_size,
        validation_steps=len(val_imgs) / batch_size,
        epochs=15,
        validation_data=validation_data,
        max_queue_size=10,
        workers=1, use_multiprocessing=False,
        verbose=1,
        callbacks=callbacks)

    print("Done!")
