flow --model cfg/yolo.cfg --load bin/yolo.weights --train --dataset "../darknet/scripts/voc/VOC2012/JPEGImages" --annotation "../darknet/scripts/voc/VOC2012/Annotations" \
--gpu 1.0 --summary "./summary" --batch 8
