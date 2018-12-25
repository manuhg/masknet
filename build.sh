python3 setup.py build_ext --inplace
sudo pip install -e .
mkdir bin

# Performance on the COCO Dataset
# Model	            mAP	    FLOPS	   FPS
# SSD300            41.2	-	       46
# SSD500            46.5	-	       19
# YOLOv2 608x608	48.1	62.94Bn    40
# Tiny YOLO         23.7	5.41 Bn	   244
# SSD321            45.4	-	       16
# DSSD321           46.1	-	       12
# R-FCN             51.9	-	       12
# SSD513            50.4	-	        8
# DSSD513           53.3	-	        6
# FPN FRCN          59.1	-	        6
# Retinanet-50-500	50.9	-	       14
# Retinanet-101-500	53.1	-	       11
# Retinanet-101-800	57.5	-	        5
# YOLOv3-320        51.5	38.97 Bn   45
# YOLOv3-416        55.3	65.86 Bn   35
# YOLOv3-608        57.9	140.69 Bn  20
# YOLOv3-tiny       33.1	5.56 Bn	   220
# YOLOv3-spp        60.6	141.45 Bn  20



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
cd ../cfg
wget -nc $yolov3_tiny_cfg
wget -nc $yolov2_tiny_cfg