## Overview

The goal of this project is to adapt [Mask R-CNN](https://github.com/matterport/Mask_RCNN) to [yolo](https://pjreddie.com/darknet/yolo/), get acceptable
quality mask while maintaining high frame rate. This solution is based on
[darkflow](https://github.com/thtrieu/darkflow), it's trained on mscoco and currently we get 88% mask
accuracy on 5K validation set. Frame rate is very high comparing to original mask r-cnn, it's 20+ fps.

## HOWTO train/run, how it works, etc.

TODO...

But the short story is:

* masknet_train.py - train on mscoco
* predict.py - predict on video file/rtsp stream
* best_weights.hdf5 - best weights so far, i.e. those with 88% accuracy

Please note that in this particular demo fps will not be too high because of video decoding, mask prediction and rendering all happen within a single thread, you'll probably get around 10fps. pure prediction however operates on 20+ fps on nVidia TITAN X.
