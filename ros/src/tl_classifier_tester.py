#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:48:45 2019

@author: benjamin
"""
import cv2
from tl_detector.light_classification.tl_classifier_yolo import TLClassifierYolo
import time 




classifier_sim = TLClassifierYolo(config_path = "./tl_detector/light_classification/config_traffic_real.json")


img = cv2.imread("/home/william/GoogleDrive/Projects/UdacityCapstone/data/left0031.jpg")




t1 = time.time()
print(classifier_sim.get_classification(img))

t2= time.time()

print("Time to process image is {:04.2f} seconds".format(t2-t1))