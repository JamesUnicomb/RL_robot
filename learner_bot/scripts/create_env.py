#!/usr/bin/env python

import rospy

from ca_msgs.msg import Bumper
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float64

import numpy as np

class createAPI:
    def __init__(self, max_lin_vel=0.2,
                       max_ang_vel=0.8):

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel

        self.reset_sub = rospy.Subscriber('create/reset', Float64, self.reset_cb)
        self.bumper_sub = rospy.Subscriber('bumper', Bumper, self.bumper_cb)
        self.wheeldrop_sub = rospy.Subscriber('wheeldrop', Empty, self.wheeldrop_cb)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)

        self.left_bumper = False
        self.right_bumper = False
        self.in_episode = False
        self.wheel_drop = False

        self.rate = rospy.Rate(30)

    def bumper_cb(self, data):
        self.left_bumper = data.is_left_pressed
        self.right_bumper = data.is_right_pressed 

    def wheeldrop_cb(self, data):
        self.wheel_drop = True

    def random_walker(self, walker_timeout):
        walker_timeout = rospy.Time.now() + rospy.Duration(walker_timeout)
        while (rospy.Time.now() < walker_timeout):
            if not self.in_episode:
                self.episode(walker_timeout)
            self.rate.sleep()
        output_cmd = Twist()
        self.cmd_vel_pub.publish(output_cmd)

    def episode(self, episode_timeout):
        self.in_episode = True
        self.wheel_drop = False
        output_cmd = Twist()
        output_cmd.linear.x = self.max_lin_vel

        sensor_test = (self.left_bumper)or(self.right_bumper)or(self.wheel_drop)

        while (not sensor_test)and(rospy.Time.now() < episode_timeout):
            sensor_test = (self.left_bumper)or(self.right_bumper)or(self.wheel_drop)
            self.cmd_vel_pub.publish(output_cmd)
            self.rate.sleep()
        
        output_cmd = Twist()
        self.cmd_vel_pub.publish(output_cmd)
        if (self.left_bumper)and(self.right_bumper):
            self.reverse()
            self.right_turn()
        elif self.left_bumper:
            self.reverse()
            self.right_turn()
        elif self.right_bumper:
            self.reverse()
            self.left_turn()
        
        self.in_episode = False       

    def reverse(self):
        action_timeout = rospy.Time.now() + rospy.Duration(0.5)
        output_cmd = Twist()
        output_cmd.linear.x = -0.2
        while (rospy.Time.now() < action_timeout):
            self.cmd_vel_pub.publish(output_cmd)
            self.rate.sleep()
        output_cmd = Twist()
        self.cmd_vel_pub.publish(output_cmd)

    def right_turn(self):
        action_timeout = rospy.Time.now() + rospy.Duration(0.2 + np.random.uniform(0.0, 1.8)) 
        output_cmd = Twist()
        output_cmd.angular.z = -self.max_ang_vel
        while (rospy.Time.now() < action_timeout):
            self.cmd_vel_pub.publish(output_cmd)
            self.rate.sleep()
        output_cmd = Twist()
        self.cmd_vel_pub.publish(output_cmd)

    def left_turn(self):
        action_timeout = rospy.Time.now() + rospy.Duration(0.2 + np.random.uniform(0.0, 1.8))
        output_cmd = Twist()
        output_cmd.angular.z = self.max_ang_vel
        while (rospy.Time.now() < action_timeout):
            self.cmd_vel_pub.publish(output_cmd)
            self.rate.sleep()
        output_cmd = Twist()
        self.cmd_vel_pub.publish(output_cmd)

    def set_velocity(self, lin, ang):
        output = Twist()
        output.linear.x = lin
        output.angular.z = ang
        self.cmd_vel_pub.publish(output)

    def reset_cb(self, timeout):
        self.random_walker(timeout.data)
