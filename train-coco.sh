#flow --model cfg/yolo-my.cfg --load bin/yolo.weights --train --dataset "../darknet/scripts/coco/images" --annotation "../darknet/scripts/blabla" \
#--gpu 1.0 --summary "./summary" --batch 8


flow --model cfg/yolo-my.cfg --load -1 --train --dataset "../darknet/scripts/coco/images" --annotation "../darknet/scripts/blabla" \
--gpu 1.0 --summary "./summary" --batch 8
