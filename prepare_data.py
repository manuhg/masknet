#! /usr/bin/env python3
import os, sys, math, time, cv2
import numpy as np
from darkflow.net.build import TFNet
from darkflow.defaults import argHandler
from multiprocessing.pool import ThreadPool
from pycocotools.coco import COCO
import tensorflow as tf

def process_coco(coco, img_path):
    res = []
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)
    processed = 0
    iter1 = 0

    imgs = imgs[:1000]

    for img in imgs:
        iter1 += 1
        processed += 1
        if iter1 > 1000:
            iter1 = 0
            print("processed", processed, len(imgs))

        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        fnd = False
        for ann in anns:
            if ('bbox' in ann) and (ann['bbox'] != []) and ('segmentation' in ann):
                fnd = True

        if fnd:
            res.append((img_path, img['file_name']))

    return res

def my_postprocess(framework, net_out, im, img_name):
    #np.save(img_name.replace('.jpg', ''), net_out)
    np.savez_compressed(img_name.replace('.jpg', ''), net_out)

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

    my_pool = ThreadPool()

    bdir = '../darknet/scripts/coco'
    all_inps = process_coco(COCO(bdir + "/annotations/person_keypoints_train2014.json"), bdir + "/images/train2014")
    all_inps += process_coco(COCO(bdir + "/annotations/person_keypoints_val2014.json"), bdir + "/images/val2014")

    batch = min(FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = my_pool.map(lambda inp: (
            np.expand_dims(tfnet.framework.preprocess(
                os.path.join(inp[0], inp[1])), 0)), this_batch)

        # Feed to the net
        feed_dict = {tfnet.inp : np.concatenate(inp_feed, 0)}
        tfnet.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = tfnet.sess.run([tfnet.out, tfnet.my_out], feed_dict)
        my_out = out[1]
        out = out[0]
        stop = time.time(); last = stop - start
        tfnet.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        tfnet.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        my_pool.map(lambda p: (lambda i, prediction:
            my_postprocess(tfnet.framework,
               prediction, os.path.join(this_batch[i][0], this_batch[i][1]), os.path.join("masknet_data", this_batch[i][1])))(*p),
            enumerate(my_out))
        stop = time.time(); last = stop - start

        # Timing
        tfnet.say('Total time = {}s / {} inps = {} ips, processed {}/{}'.format(
            last, len(inp_feed), len(inp_feed) / last, to_idx, len(all_inps)))
