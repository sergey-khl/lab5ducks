#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, String
from turbojpeg import TurboJPEG

from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from lane_follow.srv import img
from duckietown_msgs.srv import ChangePattern
import yaml
import tf.transformations as tft


# Define the HSV color range for road and stop mask
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 120, 120), (15, 255, 255)]
NUM_MASK = [(75, 190, 95), (150, 255, 255)]

# Turn pattern 
TURNS = ['S', 'S', 'L', 'R', 'S', 'R', 'L'] # starting at apriltag 3
TURN_VALUES = {'S': 0, 'L': np.pi/2, 'R': -np.pi/2}
STOP_RED = False
STOP_BLUE = False

# Set debugging mode (change to True to enable)
DEBUG = False


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
        self.proportional_stopline = None
        self.offset = 200  # 220

        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.035
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()

        self.kp_straight = 1
        self.kp_turn = 1

        # Robot Pose Variables
        self.displacement = 0
        self.orientation = 0
        
        self.subscriber = rospy.Subscriber(f'/{self.veh}/deadreckoning_node/odom', 
                                           Odometry, 
                                           self.odom_callback)

        # Initialize LED pattern change service
        led_service = f'/{self.veh}/led_controller_node/led_pattern'
        rospy.wait_for_service(led_service)
        self.led_pattern = rospy.ServiceProxy(led_service, ChangePattern)

        # Initialize get april tag service
        # april_service = f'/{self.veh}/augmented_reality_node/get_april_detect'
        # rospy.wait_for_service(april_service)
        # self.get_april = rospy.ServiceProxy(april_service, img)

        # Initialize get digit service
        #digit_service = f'/detect_digit_node/detect_digit'
        #rospy.wait_for_service(digit_service)
        #self.get_digit = rospy.ServiceProxy(digit_service, img)

        # Initialize shutdown hook
        rospy.on_shutdown(self.hook)


    def ColorMask(self, img, mask, crop_width, pid=False, stopping=False, number=False):
        global STOP_RED, STOP_BLUE
        # Convert the cropped image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply the color mask to the HSV image
        mask = cv2.inRange(hsv, mask[0], mask[1])
        crop = cv2.bitwise_and(img, img, mask=mask)
        # Find contours in the masked image
        contours_road, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

        # Find the largest contour
        max_area = 20
        max_idx = -1
        for i in range(len(contours_road)):
            area = cv2.contourArea(contours_road[i])
            if area > max_area:
                max_idx = i
                max_area = area

        # If a contour is found
        if max_idx != -1:
            # Calculate the centroid of the contour
            M = cv2.moments(contours_road[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # If using PID control, update the proportional term
                if pid:
                    self.proportional = cx - int(crop_width / 2) + self.offset

                # If checking for stopping condition or below the threshold, set STOP_RED
                elif stopping:
                    print('stoping cond cy: ', cy, cx)
                    self.proportional_stopline = (cy/170)*0.14

                    if cy >= 170 and cx in range(340, 645):
                        print('hi')
                        STOP_RED = True

                # If checking for number condition or above the threshold, set STOP_BLUE
                elif number :
                    print('number max_area: ', max_area)
                    STOP_BLUE = True

                # Draw the contour and centroid on the image (for debugging)
                if DEBUG:
                    cv2.drawContours(crop, contours_road, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        # If no contour is found, reset the flags
        else:
            if pid:
                self.proportional = None

            elif stopping:
                self.proportional_stopline = None
                STOP_RED = False

            elif number:
                STOP_BLUE = False


    def callback(self, msg):
        # Decode the JPEG image from the message
        img = self.jpeg.decode(msg.data)
        # Crop the image to focus on the region of interest
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]

        # Process the image for PID control using the ROAD_MASK
        self.ColorMask(crop, ROAD_MASK, crop_width, pid=True)
        # Process the image for stopping condition using the STOP_MASK
        self.ColorMask(crop, STOP_MASK, crop_width, stopping=True)
        # Process the image for number stopping condition using the NUM_MASK
        self.ColorMask(crop, NUM_MASK, crop_width, number=True)


    def odom_callback(self, data):
        orientation_quaternion = data.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([
            orientation_quaternion.x,
            orientation_quaternion.y,
            orientation_quaternion.z,
            orientation_quaternion.w
        ])
        self.orientation = yaw

        # Get position data
        position = data.pose.pose.position
        x, y, z = position.x, position.y, position.z

        # Calculate displacement from the origin
        displacement = np.sqrt(x**2 + y**2 + z**2)
        self.displacement = displacement


    def turn(self, r, turning_angle):
        # Get the initial orientation from the odom_callback function
        current_orientation = self.orientation

        # Calculate the target orientation (90 degrees counterclockwise)
        target_orientation = current_orientation + turning_angle
        
        orientation_error = target_orientation - current_orientation

        # Create a rospy.Rate object to maintain the loop rate at 8 Hz
        rate = rospy.Rate(8)

        while abs(orientation_error) > 0.01:
            print('orientation_error: ', orientation_error)
            orientation_error = self.target_orientation - self.orientation

            # Calculate the angular speed using a simple proportional controller
            angular_speed = self.kp_turn * orientation_error

            # Set the linear speed based on the angular speed and the desired radius
            linear_speed = angular_speed * r
            linear_speed = min(linear_speed, self.velocity)  # Limit the linear speed to the target speed

            # Set the linear and angular speeds in the Twist message
            self.twist.v = linear_speed
            self.twist.omega = angular_speed

            self.vel_pub.publish(self.twist)

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

            self.twist.omega = P + D

            if DEBUG:
                print(self.proportional, P, D, self.twist.omega, self.twist.v)

        if self.proportional_stopline is None:
            self.twist.v = self.velocity

        else:
            self.twist.v = self.velocity - self.proportional_stopline
            print('self.twist.v: ', self.twist.v)

        self.vel_pub.publish(self.twist)


    def traverse_town(self):
        global STOP_RED, STOP_BLUE
        rate = rospy.Rate(8)  # 8hz

        turn = 0

        while not rospy.is_shutdown():
            # Continue driving until a stop sign (red) or a number sign (blue) is detected
            while not STOP_RED and not STOP_BLUE:
                self.drive()
                rate.sleep()

            # Stop the Duckiebot once a sign is detected
            self.move_robust(speed=0 ,seconds=1)

            # If a stop line is detected
            if STOP_RED:
                turning_angle = TURN_VALUES[TURNS[turn]]
                if turning_angle == 0:
                    self.move_robust(speed=0.15 ,seconds=0.5)
                    # turn += 1
                    continue # go back into lane following

                else:
                    self.turn(1, turning_angle) # need to change r (will depend on which circle we are in)
                    turn += 1
                    STOP_RED = False

            # If a number sign is detected
            elif STOP_BLUE:
                # sleep? wait for number? Sergy? 
                STOP_BLUE = False
                

    def move_robust(self, speed, seconds):
        rate = rospy.Rate(10)

        self.twist.v = speed
        self.twist.omega = 0

        # Publish the twist message multiple times to ensure the robot stops
        for i in range(int(10*seconds)):
            self.vel_pub.publish(self.twist)
            rate.sleep()


    def hook(self):
        print("SHUTTING DOWN")
        self.move_robust(0)


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