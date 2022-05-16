# make sure you've built darknet before training it
cd yolov4\darknet

./darknet.exe detector train data\obj.data cfg\yolov4-custom.cfg yolov4.conv.137 -dont_show -map
