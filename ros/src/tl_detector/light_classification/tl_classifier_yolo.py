from styx_msgs.msg import TrafficLight
import os
import json 
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
from utils.utils import get_yolo_boxes
from utils.bbox import get_box_class


class TLClassifierYolo(object):
    def __init__(self,config_path):
        
        #TODO load classifier
        with open(config_path) as config_buffer:
            self.config_ = json.load(config_buffer)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config_['train']['gpus']
        self.model_ = load_model(self.config_['train']['saved_weights_name'])
        self.graph = tf.get_default_graph()
        self.net_h_, self.net_w_ = 64, 64 # a multiple of 32, the smaller the faster
        self.obj_thresh_, self.nms_thresh_ = 0.5, 0.45
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #image_rs = cv2.resize(image, (200, 150)) 
        #print("Detecting") 
        boxes = get_yolo_boxes(self.graph,self.model_, [image], self.net_h_, self.net_w_, self.config_['model']['anchors'], self.obj_thresh_, self.nms_thresh_)[0]
        ccc = get_box_class(boxes, self.config_['model']['labels'], self.obj_thresh_, quiet=True)
        
        
        if(ccc == [[], []]):
            return TrafficLight.UNKNOWN
        elif (ccc[0][np.argmax(ccc[1])] == "Red"):
            return(TrafficLight.RED)
        else:
            return(TrafficLight.GREEN)
     
        
      
        
        
