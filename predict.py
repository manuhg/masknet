#! /usr/bin/env python3
import os, sys, math, time, cv2
import numpy as np
from darkflow.net.build import TFNet
from darkflow.defaults import argHandler
from multiprocessing.pool import ThreadPool
from pycocotools.coco import COCO
import tensorflow as tf
import masknet
from time import time as timer
from scipy.misc import imresize

def my_postprocess(framework, net_out, im):
    boxes = framework.findboxes(net_out)

    # meta
    meta = framework.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    imgcv = im
    h, w, _ = imgcv.shape

    rois = []

    for b in boxes:
        boxResults = framework.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)

        if (mess != "person"):
            continue

        x1 = np.float32(left / w)
        y1 = np.float32(top / h)
        x2 = np.float32(right / w)
        y2 = np.float32(bot / h)

        assert(len(rois) < masknet.my_num_rois)

        rois.append([y1, x1, y2, x2])

        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            colors[max_indx], thick)
        cv2.putText(imgcv, mess, (left, top - 12),
            0, 1e-3 * h, colors[max_indx],thick//3)

    num_true_rois = len(rois)

    for _ in range(masknet.my_num_rois - len(rois)):
        rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])

    return np.array(rois), num_true_rois

if __name__ == "__main__":
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(sys.argv)

    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)

    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir, 'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)
    tfnet = TFNet(FLAGS)

    model = masknet.create_model()
    model.summary()
    model.load_weights("weights.hdf5")

    file = FLAGS.demo
    SaveVideo = True

    if file == 'camera':
        file = 0
    else:
        pass
        #assert os.path.isfile(file), \
        #'file {} does not exist'.format(file)

    camera = cv2.VideoCapture(file)

    if file == 0:
        tfnet.say('Press [ESC] to quit demo')

    assert camera.isOpened(), \
    'Cannot capture source'

    file = 0
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('', width, height)
    else:
        _, frame = camera.read()
        height, width, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            'video.avi', fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = int()
    start = timer()
    tfnet.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        preprocessed = tfnet.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        # Only process and imshow when queue is full
        if elapsed % FLAGS.queue == 0:
            feed_dict = {tfnet.inp: buffer_pre}
            net_out = tfnet.sess.run([tfnet.out, tfnet.my_c2, tfnet.my_c3, tfnet.my_c4, tfnet.my_c5], feed_dict)
            my_c2 = net_out[1]
            my_c3 = net_out[2]
            my_c4 = net_out[3]
            my_c5 = net_out[4]
            net_out = net_out[0]
            for img, single_out, c2, c3, c4, c5 in zip(buffer_inp, net_out, my_c2, my_c3, my_c4, my_c5):
                rois, num_true_rois = my_postprocess(tfnet.framework, single_out, img)

                c2 = np.array(c2)
                c2 = c2[np.newaxis, ...]

                c3 = np.array(c3)
                c3 = c3[np.newaxis, ...]

                c4 = np.array(c4)
                c4 = c4[np.newaxis, ...]

                c5 = np.array(c5)
                c5 = c5[np.newaxis, ...]

                inp2 = np.array(rois)
                inp2 = inp2[np.newaxis, ...]

                p = model.predict([c2, c3, c4, c5, inp2])

                p = p[0, :num_true_rois, :, :, 0]

                frame_h, frame_w, _ = img.shape

                for i in range(num_true_rois):
                    roi = rois[i]
                    mask = p[i]

                    y1, x1, y2, x2 = roi

                    left = int(x1 * frame_w)
                    top = int(y1 * frame_h)
                    right = int(x2 * frame_w)
                    bot = int(y2 * frame_h)

                    mask = imresize(mask, (bot - top, right - left), interp='bilinear').astype(np.float32) / 255.0
                    mask2 = np.where(mask >= 0.5, 1, 0).astype(np.uint8)

                    if (i % 3) == 0:
                        mask3 = cv2.merge((mask2 * 0, mask2 * 0, mask2 * 255))
                    elif (i % 3) == 1:
                        mask3 = cv2.merge((mask2 * 0, mask2 * 255, mask2 * 0))
                    else:
                        mask3 = cv2.merge((mask2 * 255, mask2 * 0, mask2 * 0))

                    img[top:bot,left:right] = cv2.addWeighted(img[top:bot,left:right], 1.0, mask3, 0.8, 0)

                videoWriter.write(img)
                cv2.imshow('', img)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0: #camera window
            k = cv2.waitKey(1) & 0xff
            if k == 32:
                k = cv2.waitKey() & 0xff
            if k == 27:
                break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0: #camera window
        cv2.destroyAllWindows()
