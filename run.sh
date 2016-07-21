#!/bin/bash
# Execute classify with standard test input.

./classify.py /home/gabriel/tmp/1_auto_select/ 224 /home/gabriel/caffe_models/GNet_X_Dataset_V3/googlenet_x_dataset_v3_augmentation_iter_132000.caffemodel 0 30 /home/gabriel/caffe_models/GNet_X_Dataset_V3/deploy.prototxt layer1,layer2,layer3 /home/gabriel/outputs/
