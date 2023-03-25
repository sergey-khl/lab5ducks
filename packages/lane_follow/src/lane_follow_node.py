#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, String
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Point
from lane_follow.srv import img
from duckietown_msgs.srv import ChangePattern
import yaml
from nav_msgs.msg import Odometry

# Define the HSV color range for road and stop mask
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 120, 120), (15, 255, 255)]
NUM_MASK = [(75, 190, 95), (150, 255, 255)]

# Turn pattern 
PATH = ['S', 'S', 'L', 'R', 'S', 'R', 'L'] # starting at apriltag 3
STOP_RED = False
STOP_BLUE = False

# If using PID controller for collision avoidance
PID_COLLISION = False 

# Set debugging mode (change to True to enable)
DEBUG = False

# Determines what side of the road the robot drives on (change to True to dirve on left)
ENGLISH = False

class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Save node name and vehicle name
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.sub_pose = rospy.Subscriber(f'/{self.veh_name}/deadreckoning_node/odom', 
                                    Odometry, self.odom_callback)
        

        # Initialize distance subscriber and velocity publisher
        #self.sub_ml = rospy.Subscriber("/" + self.veh + "/augmented_reality_node/position", Point, self.cb_april, queue_size=1)
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        self.calibration_file = f'/data/config/calibrations/camera_intrinsic/default.yaml'
 
        self.calibration = self.readYamlFile(self.calibration_file)

        self.img_width = self.calibration['image_width']
        self.img_height = self.calibration['image_height']
        self.cam_matrix = np.array(self.calibration['camera_matrix']['data']).reshape((self.calibration['camera_matrix']['rows'], self.calibration['camera_matrix']['cols']))
        self.distort_coeff = np.array(self.calibration['distortion_coefficients']['data']).reshape((self.calibration['distortion_coefficients']['rows'], self.calibration['distortion_coefficients']['cols']))

        self.new_cam_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.distort_coeff, (self.img_width, self.img_height), 1, (self.img_width, self.img_height))
        
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()
        self.undistorted = None

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220

        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.035
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Initialize LED pattern change service
        led_service = f'/{self.veh}/led_controller_node/led_pattern'
        rospy.wait_for_service(led_service)
        self.led_pattern = rospy.ServiceProxy(led_service, ChangePattern)

        # Initialize get april tag service
        april_service = f'/{self.veh}/augmented_reality_node/get_april_detect'
        rospy.wait_for_service(april_service)
        self.get_april = rospy.ServiceProxy(april_service, img)

        # Initialize get digit service
        #digit_service = f'/detect_digit_node/detect_digit'
        #rospy.wait_for_service(digit_service)
        #self.get_digit = rospy.ServiceProxy(digit_service, img)

        # Initialize shutdown hook
        rospy.on_shutdown(self.hook)

    def ColorMask(self, msg, mask, pid=False, stopping=False, number=False):
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, mask[0], mask[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours_road, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        
        max_area = 20
        max_idx = -1
        for i in range(len(contours_road)):
            area = cv2.contourArea(contours_road[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours_road[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if pid:
                    self.proportional = cx - int(crop / 2) + self.offset

                elif stopping or cy < 140: # need to tune parameter / and change to or
                    print('stoping cond cy: ', cy) 
                    STOP_RED = True 

                elif number or max_area > 50: # need to tune parameter / and change to or
                    print('number max_area: ', max_area) 
                    STOP_BLUE = True 

                if DEBUG:
                    cv2.drawContours(crop, contours_road, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        else:
            if pid:
                self.proportional = None

            elif stopping: 
                STOP_RED = False 

            elif number: 
                STOP_BLUE = False 


    def callback(self, msg):
        self.ColorMask(msg, ROAD_MASK, pid=True)
        self.ColorMask(msg, STOP_MASK, stopping=True)
        self.ColorMask(msg, NUM_MASK, number=True)


    def turn(self, v, omega, hz, delay):
        # Set the rate at which to publish the twist messages
        rate = rospy.Rate(hz)
        # Set the delay before the next movement
        self.delay = delay

        # Set the linear velocity of the robot
        self.twist.v = v

        # Set the angular velocity of the robot
        self.twist.omega = omega

        # Publish the twist message multiple times to ensure the robot turns
        for i in range(8):
            self.vel_pub.publish(self.twist)

        # Sleep for the desired time to allow the robot to turn
        rate.sleep()
                

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
            self.last_error = 0

        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D
            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        self.vel_pub.publish(self.twist)


    def traverse_town(self):
        rate = rospy.Rate(8)  # 8hz

        turn = 0

        while not rospy.is_shutdown():
            while not STOP_RED and not STOP_BLUE:
                self.drive()
                rate.sleep()

            if STOP_RED:
                self.turn(PATH[turn], inner=False, outer=False)
                # need to sleep?
                STOP_RED = False

            elif STOP_BLUE:
                # sleep? wait for number?  
                STOP_BLUE = False
                pass
    

    def hook(self):
        print("SHUTTING DOWN")
        # have 1 delay to avoid 0 division error
        rate = rospy.Rate(10)

        # Set the linear velocity of the robot
        self.twist.v = 0

        # Set the angular velocity of the robot
        self.twist.omega = 0

        # Publish the twist message multiple times to ensure the robot turns
        for i in range(20):
            self.vel_pub.publish(self.twist)

            # Sleep for the desired time to allow the robot to die
            rate.sleep()


    def change_led_lights(self, dir: str):
        '''
        Sends msg to service server
        ignore
        Colors:
            "off": [0,0,0],
            "white": [1,1,1],
            "green": [0,1,0],
            "red": [1,0,0],
            "blue": [0,0,1],
            "yellow": [1,0.8,0],
            "purple": [1,0,1],
            "cyan": [0,1,1],
            "pink": [1,0,0.5],
        '''
        msg = String()
        msg.data = dir
        self.led_pattern(msg)
        

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


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    node.traverse_town()