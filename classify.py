#!/usr/bin/python
import cv2
import argparse
import caffe
import os
import numpy as np
import sys
import time
from multiprocessing import Process, Queue, current_process, freeze_support

class SmartFeaturesExtractor():
    def __init__(self):
        self.running = True
    
    def create_network(self, model_path, arch_path, device):
        caffe.set_mode_gpu()
        caffe.set_device(device)
        self.net = caffe.Net(arch_path,model_path,caffe.TEST)
    
    def has_processes(self):
        for p in self.processes:
            if p.is_alive():
                return True
        return False

    def terminate_processes(self):
        self.running = False
        for p in self.processes:
            if p.is_alive():
                p.terminate()

    def predict(self, base, input_size, batch_size):
        files = os.listdir(base)        

        task_queue = Queue()
        done_queue = Queue()
        for image_name in files:
            task_queue.put(image_name)
    
        NUMBER_OF_PROCESSES = 6

        self.processes = [0] * NUMBER_OF_PROCESSES
        for i in range(NUMBER_OF_PROCESSES):
            self.processes[i] = Process(target=self.image_reader, args=(base, input_size, task_queue, done_queue))
            self.processes[i].start()
            print "Process", i

        while not task_queue.empty() or not done_queue.empty():
            print "Get images from done queue..."
            while done_queue.qsize() < batch_size and self.has_processes():
                print "Waitting for image buffer...", done_queue.qsize()
                time.sleep(1)

            batch = []
            print "Task Queue:", task_queue.qsize()
            print "Done Queue:", done_queue.qsize()
            while done_queue.qsize() > 0 and len(batch) < batch_size:
                batch += [done_queue.get()]

            self.net.blobs['data'].reshape(len(batch),3,input_size,input_size)
            for i, im_data in enumerate(batch):            
                self.net.blobs['data'].data[i,:,:,:] = im_data[1]

            print "Pred..."
            out = self.net.forward()            
            for i, im_data in enumerate(batch):            
                print im_data[0], out['prob'][i]            

    def image_reader(self, base, input_size, input_tasks, output_task):
        means = np.asarray([116,136,169]).astype(np.float32)
        for im_name in iter(input_tasks.get, 'STOP'):
            im_data = cv2.imread(os.path.join(base, im_name)).astype(np.float32)
            im_data = SmartFeaturesExtractor.resize_image(im_data, input_size)
            im_data -= means
            im_data = im_data.transpose(2,0,1)[np.newaxis,:,:,:]
            while output_task.qsize() > 500 and self.running:
                time.sleep(1)
            output_task.put((im_name, im_data))

    @staticmethod
    def resize_image(img, new_min_dim):
        width = int(img.shape[1])
        height = int(img.shape[0])
        new_width = 0
        crop = 0

        if width < height:
            new_width = new_min_dim
            new_height = (new_min_dim * height) / width
            crop = new_height - new_min_dim
            img = cv2.resize(img, (new_width, new_height), 0, 0, cv2.INTER_CUBIC)
            img = img[crop / 2:new_min_dim + (crop / 2),:]
        else:
            new_height = new_min_dim
            new_width = (new_min_dim * width) / height
            crop = new_width - new_min_dim        
            img = cv2.resize(img, (new_width, new_height), 0, 0, cv2.INTER_CUBIC)
            img = img[:,crop / 2:new_min_dim + (crop / 2)]
        return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("IMAGES_PATH",          help="Path to the input images.")
    ap.add_argument("IMAGE_SIZE",           help="Square size for input images.")
    ap.add_argument("MODEL_PATH",           help="Path to the caffemodel file.")
    ap.add_argument("GPU_DEVICE",           help="GPU device ID.")
    ap.add_argument("BATCH_SIZE",           help="Size of data batch.")
    ap.add_argument("ARCH_PATH",            help="Path to the caffe network architecture.")
    ap.add_argument("OUTPUT_LAYERS",        help="Comma separated [layerA,layerB,layerC] list of features extracted layers.")
    ap.add_argument("FEATURES_OUTPUT_PATH", help="Path to the extracted images features.")
    
    args = ap.parse_args()    
    out_layers = args.OUTPUT_LAYERS.split(',')    

    sfs = SmartFeaturesExtractor()
    try:
        sfs.create_network(args.MODEL_PATH, args.ARCH_PATH, int(args.GPU_DEVICE))
        sfs.predict(args.IMAGES_PATH, int(args.IMAGE_SIZE), int(args.BATCH_SIZE))
    except:
        print "EXCEPTION!", sys.exc_info()[1]        
    sfs.terminate_processes()
