#!/usr/bin/env python3
import os
from pathlib import Path
import rospy
import cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from turbojpeg import TurboJPEG
import yaml
from geometry_msgs.msg import Quaternion, Pose, Point
from lane_follow.srv import img

from tf2_ros import TransformBroadcaster, Buffer, TransformListener

from tf import transformations as tr

class DigitVirtuosoNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(DigitVirtuosoNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_param("~veh")
       
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")

        # setup publisher
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()
        self.undistorted = None

        self.calibration_file = f'/data/config/calibrations/camera_intrinsic/default.yaml'
 
        self.calibration = self.readYamlFile(self.calibration_file)

        self.img_width = self.calibration['image_width']
        self.img_height = self.calibration['image_height']
        self.cam_matrix = np.array(self.calibration['camera_matrix']['data']).reshape((self.calibration['camera_matrix']['rows'], self.calibration['camera_matrix']['cols']))
        self.distort_coeff = np.array(self.calibration['distortion_coefficients']['data']).reshape((self.calibration['distortion_coefficients']['rows'], self.calibration['distortion_coefficients']['cols']))

        self.new_cam_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.distort_coeff, (self.img_width, self.img_height), 1, (self.img_width, self.img_height))
        
        # Initialize get april tag service
        april_service = f'/{self.veh}/augmented_reality_node/get_april_detect'
        rospy.wait_for_service(april_service)
        self.get_april = rospy.ServiceProxy(april_service, img)

        # Initialize get digit service
        digit_service = f'/detect_digit_node/detect_digit'
        rospy.wait_for_service(digit_service)
        self.get_digit = rospy.ServiceProxy(digit_service, img)

    def callback(self, msg):
        # decode the received message to an image
        img = self.jpeg.decode(msg.data)

        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        undistorted = cv2.undistort(img, self.cam_matrix, self.distort_coeff, None, self.new_cam_matrix)
        x, y, w, h = self.roi
        undistorted = undistorted[y:y+h, x:x+w]
        self.undistorted = CompressedImage(format="jpeg", data=cv2.imencode('.jpg', undistorted)[1].tobytes())

    def maestro(self):
        if self.undistorted is not None:
            num = self.get_digit(self.undistorted)
            print(num.found.data)
            detected = self.get_april(self.undistorted)
            print(detected.found.data)

        
    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                        %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == '__main__':
    # create the node
    node = DigitVirtuosoNode(node_name='digit_virtuoso_node')
    rate = rospy.Rate(1)  # 8hz
    while not rospy.is_shutdown():
        node.maestro()
        rate.sleep()