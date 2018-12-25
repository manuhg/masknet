cd bin
yolo_small="https://pjreddie.com/media/files/yolo-small.weights" #YOLOv1
yolo="https://pjreddie.com/media/files/yolo.weights" #YOLOv2
yolov2="https://pjreddie.com/media/files/yolov2.weights"
yolov3="https://pjreddie.com/media/files/yolov3.weights"
yolov2_tiny="https://pjreddie.com/media/files/yolov2-tiny.weights"
yolov2_tiny_cfg='https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg'
yolov3_tiny_cfg='https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
yolov3_tiny_weigths='https://pjreddie.com/media/files/yolov3-tiny.weights'
strided_cfg='https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/strided.cfg'
strided_weigths='https://pjreddie.com/media/files/strided.weights'
wget -nc $yolo_small
wget -nc $yolo
wget -nc $yolov2
wget -nc $yolov3
wget -nc $yolov2_tiny
wget -nc $yolov3_tiny_weigths
wget -nv $strided_weigths
cd ../cfg
wget -nc $yolov3_tiny_cfg
wget -nc $yolov2_tiny_cfg
wget -nv $strided_cfg
