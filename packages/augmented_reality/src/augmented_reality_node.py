#!/usr/bin/env python3
import os
from pathlib import Path
import rospy
import cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from turbojpeg import TurboJPEG
import yaml
from geometry_msgs.msg import Quaternion, Pose, Point, TransformStamped, Vector3, Transform
from lane_follow.srv import img, imgResponse
from dt_apriltags import Detector

from tf2_ros import TransformBroadcaster, Buffer, TransformListener

from tf import transformations as tr

class AugmentedRealityNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(AugmentedRealityNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_param("~veh")
       
        # setup april tag detector
        self.detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
                       
        self.tags = {
        "153": {"x": 1.75, "y": 1.252, "z": 0.075, "yaw": 0, "pitch": 0, "roll": 4.7124},
        "201": {"x": 1.65, "y": 0.17, "z": 0.075, "yaw": 3.92699, "pitch": 0, "roll": 4.7124},
        "200": {"x": 0.17, "y": 0.17, "z": 0.075, "yaw": 2.3562, "pitch": 0, "roll": 4.7124},
        "162": {"x": 1.253, "y": 1.253, "z": 0.075, "yaw": 4.7124, "pitch": 0, "roll": 4.7124},
        "58": {"x": 0.574, "y": 1.259, "z": 0.075, "yaw": 4.7124, "pitch": 0, "roll": 4.7124},
        "133": {"x": 1.253, "y": 1.755, "z": 0.075, "yaw": 3.14159265, "pitch": 0, "roll": 4.7124},
        "169": {"x": 0.574, "y": 1.755, "z": 0.075, "yaw": 1.5708, "pitch": 0, "roll": 4.7124},
        "62": {"x": 0.075, "y": 1.755, "z": 0.075, "yaw": 3.141592, "pitch": 0, "roll": 4.7124},
        "94": {"x": 1.65, "y": 2.84, "z": 0.075, "yaw": 5.49779, "pitch": 0, "roll": 4.7124},
        "93": {"x": 0.17, "y": 2.84, "z": 0.075, "yaw": 0.7854, "pitch": 0, "roll": 4.7124},
        }

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

        # Services
        self.srv_get_april = rospy.Service(
            "~get_april_detect", img, self.srvGetApril
        )



    def srvGetApril(self, req):
        undistorted = self.jpeg.decode(req.img.data)
        self.undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(self.undistorted, (28, 28))

        detected = self.detect_april()

        return imgResponse(Int32(data=detected))


    def detect_april(self):
        # https://pyimagesearch.com/2020/11/02/apriltag-with-python/
        # https://github.com/duckietown/lib-dt-apriltags/blob/master/test/test.py
        # april tag detection
        
        results = self.detector.detect(self.undistorted, estimate_tag_pose=True, camera_params=(self.cam_matrix[0,0], self.cam_matrix[1,1], self.cam_matrix[0,2], self.cam_matrix[1,2]), tag_size=0.065)

        try:
            r = results[0]

            self.log(f"\napriltag ID: {r.tag_id}, \nlocation: {self.tags[str(r.tag_id)]['x']}, {self.tags[str(r.tag_id)]['y']}, {self.tags[str(r.tag_id)]['z']}, \nrotation: {self.tags[str(r.tag_id)]['yaw']}, {self.tags[str(r.tag_id)]['pitch']}, {self.tags[str(r.tag_id)]['roll']}\n")

            return 1
        except Exception as e:
            print(e)

            return 0
        
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
    node = AugmentedRealityNode(node_name='augmented_reality_node')

    rospy.spin()
