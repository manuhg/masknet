python3 setup.py build_ext --inplace
sudo pip install -e .
mkdir bin

cd bin
yolo_small = "https://pjreddie.com/media/files/yolo-small.weights" #YOLOv1
yolo = "https://pjreddie.com/media/files/yolo.weights" #YOLOv2
tiny_yolo_voc = "https://pjreddie.com/media/files/tiny-yolo-voc.weights" #YOLOv2
wget $yolo_small
wget $yolo
wget $tiny_yolo_voc