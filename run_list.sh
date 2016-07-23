#!/bin/bash
# Execute classify with standard test input.

./classify.py /media/gabriel/data_doc/X_Dataset_V3/ /media/gabriel/data_doc/X_Dataset_V3/index/index_x_dataset_v3_test.txt 224 /home/gabriel/caffe_models/GNet_X_Dataset_V3/googlenet_x_dataset_v3_augmentation_iter_132000.caffemodel 0 50 /home/gabriel/caffe_models/GNet_X_Dataset_V3/deploy.prototxt layer1,layer2,layer3 /home/gabriel/outputs/
