#!/usr/bin/env python3
import os
from pathlib import Path
import rospy
import cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose, Point, TransformStamped, Vector3, Transforms
from lane_follow.srv import img
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


        # setup publisher
        self.undistorted = None

        # construct publisher
        self.pub_loc = rospy.Publisher(f'/{self.veh}/teleport', Pose, queue_size=1)

        # Services
        self.srv_get_april = rospy.Service(
            "~get_april_detect", img, self.srvGetApril
        )


        self._tf_broadcaster = TransformBroadcaster()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self.wizard(Point(*[0.32, 0.3, 0]), Quaternion(*[0, 0, 0, 1]))

    def srvGetApril(self, req):
        print('got undistorted')
        undistorted = req.img.data
        self.undistorted = cv2.imdecode(undistorted, 0) 
        
        detected = self.detect_april()

        return


    def detect_april(self):
        # https://pyimagesearch.com/2020/11/02/apriltag-with-python/
        # https://github.com/duckietown/lib-dt-apriltags/blob/master/test/test.py
        # april tag detection
        
        results = self.detector.detect(self.undistorted, estimate_tag_pose=True, camera_params=(self.cam_matrix[0,0], self.cam_matrix[1,1], self.cam_matrix[0,2], self.cam_matrix[1,2]), tag_size=0.065)

        # for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        try:
            r = results[0]
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(self.undistorted, ptA, ptB, (0, 255, 0), 2)
            cv2.line(self.undistorted, ptB, ptC, (0, 255, 0), 2)
            cv2.line(self.undistorted, ptC, ptD, (0, 255, 0), 2)
            cv2.line(self.undistorted, ptD, ptA, (0, 255, 0), 2)
            
            # draw the tag id on the image center
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            tagID = str(r.tag_id)
            cv2.putText(self.undistorted, tagID, (cX - 15, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            t = tr.translation_from_matrix(tr.translation_matrix(np.array(r.pose_t).reshape(3)))

            q = np.append(r.pose_R, [[0],[0],[0]], axis =1)
            q = np.append(q, [[0,0,0, 1]], axis =0)
            q = tr.quaternion_from_matrix(q)

            # send predicted april tag position
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()  # Ideally, should be encoder time
            odom.header.frame_id = f"{self.veh}/camera_optical_frame"

            self._tf_broadcaster.sendTransform(
                TransformStamped(
                    header=odom.header,
                    child_frame_id=f"{self.veh}/at_{r.tag_id}_estimate",
                    transform=Transform(
                        translation=Vector3(*t), rotation=Quaternion(*q)
                    ),
                )
            )


            trans = self._tf_buffer.lookup_transform(f"{self.veh}/world", f"{self.veh}/at_{r.tag_id}_estimate", rospy.Time())

            print(f"apriltag ID: {r.tag_id}, \nlocation: {trans.transform.translation.x}, {trans.transform.translation.y}, {0.5}")

            #new_img = CompressedImage()
            #new_img.data = cv2.imencode('.jpg', self.undistorted)[1].tobytes()
            #self.pub_img.publish(new_img)

            # if (r.tag_id == 93 or r.tag_id == 94 or r.tag_id == 200 or r.tag_id == 201):
            #     self.log('UOFA')
            #     self.change_led_lights("green")
            # elif (r.tag_id == 62 or r.tag_id == 153 or r.tag_id == 133 or r.tag_id == 56):
            #     self.log('INTERSECTION')
            #     self.change_led_lights("blue")
            # elif (r.tag_id == 162 or r.tag_id == 169):
            #     self.log('STOP')
            #     self.change_led_lights("red")
            


            # may not need

            # find transform from april tag to wheelbase in worldframe
            # https://github.com/ros/geometry2/blob/noetic-devel/tf2_ros/src/tf2_ros/buffer.py
            trans = self._tf_buffer.lookup_transform_full(f"{self.veh}/at_{r.tag_id}_estimate", rospy.Time(), f"{self.veh}/base", rospy.Time(), f"{self.veh}/world")
            
            # teleport
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()  # Ideally, should be encoder time
            odom.header.frame_id = f"{self.veh}/at_{r.tag_id}_static"

            self._tf_broadcaster.sendTransform(TransformStamped(
                    header=odom.header,
                    child_frame_id=f"{self.veh}/robo_estimate",
                    transform=trans.transform,
                )
            )


            trans = self._tf_buffer.lookup_transform(f"{self.veh}/world", f"{self.veh}/robo_estimate", rospy.Time())

            

            self.wizard(Point(*[trans.transform.translation.x, trans.transform.translation.y, 0]), trans.transform.rotation)
            return True
        except Exception as e:
            print(e)
            self.change_led_lights("white")
            return False

    def wizard(self, tran, rot):
        pose = Pose(tran, rot)
        self.pub_loc.publish(pose)

if __name__ == '__main__':
    # create the node
    node = AugmentedRealityNode(node_name='augmented_reality_node')

    rospy.spin()
