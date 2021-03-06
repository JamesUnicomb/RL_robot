#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class RLData:
    def __init__(self):
        self.bridge = CvBridge()

        self.depth_image = np.zeros((640, 480))
        self.laser_array = np.zeros(640)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0

        depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)
        laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_cb)
        odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb)


    def depth_cb(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image)
            self.depth_image = cv_image
        except CvBridgeError as e:
            print(e)


    def laser_cb(self, laser):
        ranges = np.array(laser.ranges)[1::4]
        ranges[ranges == np.nan] = 0.0
        self.laser_array = ranges


    def odom_cb(self, odometry):
        self.odom_x = odometry.pose.pose.position.x
        self.odom_y = odometry.pose.pose.position.y


    def cmd_cb(self, cmd_vel):
        self.lin_vel = cmd_vel.linear.x
        self.ang_vel = cmd_vel.angular.z
        
