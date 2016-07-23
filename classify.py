#!/usr/bin/python
import cv2
import argparse
import caffe
import os
import numpy as np
import sys
import time
from multiprocessing import Process, Queue, current_process, freeze_support
from sklearn.metrics import accuracy_score
from abc import ABCMeta, abstractmethod

class SmartFeaturesExtractor():
    def __init__(self):
        # self.running = True
        self.job_maker = None
    
    def set_job_maker(self, job_maker):
        self.job_maker = job_maker

    def create_network(self, model_path, arch_path, device):
        caffe.set_mode_gpu()
        caffe.set_device(device)
        self.net = caffe.Net(arch_path,model_path,caffe.TEST)

    def terminate_processes(self):
        self.running = False        
        for p in self.processes:
            if p.is_alive():
                p.terminate()

        self.done_queue.close()
        self.task_queue.close()        

    @staticmethod
    def load_input_list(images_path, index_path):
        if os.path.isdir(index_path):
            return (index_path, os.listdir(index_path))
        with open(index_path, "r") as f:
            lines = f.readlines()
        lines = [(line.strip().split(" ")[0], int(line.strip().split(" ")[1])) for line in lines]
        X, y = map(list,zip(*lines))
        return (images_path, X, y)

    def forward(self, input_data, input_size, batch_size):
        base_path = input_data[0]
        X = input_data[1]
        Y = input_data[2] if len(input_data) == 3 else [None] * len(X)
        X_pred = []
        self.task_queue = Queue()
        self.done_queue = Queue()
        for x,y in zip(X,Y):
            self.task_queue.put((x,y))
    
        NUMBER_OF_PROCESSES = 4

        self.processes = [0] * NUMBER_OF_PROCESSES
        for i in range(NUMBER_OF_PROCESSES):
            self.processes[i] = Process(target=self.image_reader, args=(base_path, input_size, self.task_queue, self.done_queue))
            self.processes[i].start()
            print "Process", i

        while not self.task_queue.empty() or not self.done_queue.empty():
            while self.done_queue.qsize() < batch_size and self.task_queue.qsize() > 0:
                time.sleep(1)

            batch_name = []
            batch_y = []
            if self.done_queue.qsize() < batch_size:
                self.net.blobs['data'].reshape(self.done_queue.qsize(),3,input_size,input_size)
            else:
                self.net.blobs['data'].reshape(batch_size,3,input_size,input_size)
            batch_index = 0
            while self.done_queue.qsize() > 0 and batch_index < batch_size:
                data = self.done_queue.get()
                batch_name += [data[0]]                
                self.net.blobs['data'].data[batch_index,:,:,:] = data[1]
                batch_y += [data[2]]
                batch_index += 1

            out = self.net.forward()
            self.job_maker.do_the_job({"net":self.net, "file_name":batch_name, "y_pred":batch_y, "y":Y})            
        
    def image_reader(self, base, input_size, input_tasks, output_task):
        means = np.asarray([116,136,169]).astype(np.float32)
        while not input_tasks.empty():
            task = input_tasks.get()
            im_name = task[0]
            im_class = task[1]
            im_data = cv2.imread(os.path.join(base, task[0])).astype(np.float32)
            im_data = SmartFeaturesExtractor.resize_image(im_data, input_size)            
            im_data -= means
            im_data = im_data.transpose(2,0,1)[np.newaxis,:,:,:]
            while output_task.qsize() >= 500:
                time.sleep(1)                
            output_task.put((im_name, im_data, im_class))

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

class AJob:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def do_the_job(self, data_dic):
        raise NotImplementedError()

class SaveFeatures(AJob):
    def __init__(self):
        self.y_batch = []
        self.y_pred = []

    def do_the_job(self, data_dic):
        file_names = data_dic["file_name"]
        net = data_dic["net"]
        self.y_pred += net.blobs['prob'].data.argmax(axis=1).tolist()
        self.y_batch += data_dic["y_pred"]
        y_real = data_dic["y"]
        for i, im_data in enumerate(file_names):            
            print file_names[i], net.blobs['prob'].data[i]
        if len(self.y_batch) != len(y_real):
            return
        print "Final ACC:", accuracy_score(self.y_pred, self.y_batch)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("IMAGES_PATH",          help="Path to the input images.")
    ap.add_argument("IMAGES_INDEX",         help="Path to the input images.")
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
    sfs.set_job_maker(SaveFeatures())
    try:
        sfs.create_network(args.MODEL_PATH, args.ARCH_PATH, int(args.GPU_DEVICE))
        sfs.forward(sfs.load_input_list(args.IMAGES_PATH, args.IMAGES_INDEX), int(args.IMAGE_SIZE), int(args.BATCH_SIZE))
    except:
        print "EXCEPTION:"
        for e in sys.exc_info():
            print e
    sfs.terminate_processes()
    sys.exit(0)
