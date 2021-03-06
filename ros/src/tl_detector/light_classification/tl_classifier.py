from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        result = TrafficLight.UNKNOWN
       
        red = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        l_red = np.array([0,50,50])
        h_red = np.array([10,255,255])
        red1 = cv2.inRange(red, l_red , h_red)


        l_red = np.array([170,50,50])
        h_red = np.array([180,255,255])
        red2 = cv2.inRange(red, l_red , h_red)

        converted_img = cv2.addWeighted(red1, 1.0, red2, 1.0, 0.0)

        blur_img = cv2.GaussianBlur(converted_img,(15,15),0)


        circles = cv2.HoughCircles(blur_img,cv2.HOUGH_GRADIENT,0.5,41, param1=70,param2=30,minRadius=15,maxRadius=150)

        found = False 
        if circles is not None:
            result = TrafficLight.RED
        
      
        
        
        return result