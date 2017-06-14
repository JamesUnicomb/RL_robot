#!/usr/bin/env python
import sys, time

import numpy as np

import rospy
from geometry_msgs.msg import Twist

accel_lim_x = 0.35
accel_lim_w = 0.65

rospy.init_node('cmd_vel_smoother', anonymous=True)

class velocity_multiplexer:
    def __init__(self):
        self.vel_pub = rospy.Publisher('create/cmd_vel', Twist, queue_size = 10)
        self.vel_seb = rospy.Subscriber('cmd_vel', Twist, self.callback)

        now = rospy.Time.now()
        self.start_time = now.secs + now.nsecs * 10 ** -9

    def callback(self, data):
        self.input_lin_vel = data.linear.x
        self.input_ang_vel = data.angular.z
        
        now = rospy.Time.now()
        self.last_publish_time = now.secs + now.nsecs * 10 ** -9

    def time_difference(self):
        now = rospy.Time.now()
        current_time = now.secs + now.nsecs * 10 ** -9
        rospy.loginfo('time running: %f' % (current_time - self.start_time))

def main(args):
    #initialise the space of variables
    nu_output = 0.0
    om_output = 0.0

    #call the class used for contructing the problem
    vm = velocity_multiplexer()
 
    #start a loop to make the multiplexer continuous
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        #the time differential is outside of control loop. 
        try:
            now
        except UnboundLocalError:
            dt = 1.0/10.0
        else:
            dt = (rospy.Time.now().secs + rospy.Time.now().nsecs * 10 ** -9) - (now.secs + now.nsecs * 10 ** -9)

        now = rospy.Time.now()
        try:
            vm.last_publish_time
        except AttributeError:
            output_msg = Twist()
            vm.vel_pub.publish(output_msg)
        else:
            output_msg = Twist()
            if (((now.secs + now.nsecs * 10 ** -9) - vm.last_publish_time) > 0.35):
                target_lin_vel, target_ang_vel = (0.0, 0.0)
            else:
                target_lin_vel, target_ang_vel = (vm.input_lin_vel, vm.input_ang_vel)


            if (nu_output != target_lin_vel) or (om_output != target_ang_vel):
                if abs(target_lin_vel - nu_output) > accel_lim_x * dt:
                    nu_output += np.sign(target_lin_vel - nu_output) * accel_lim_x * dt
                else:
                    nu_output = target_lin_vel

                if abs(target_ang_vel - om_output) > accel_lim_w * dt:
                    om_output += np.sign(target_ang_vel - om_output) * accel_lim_w * dt
                else:
                    om_output = target_ang_vel

            else:
                nu_output = target_lin_vel
                om_output = target_ang_vel

            # simple filter to omit small values.
            output_msg.linear.x = nu_output 
            output_msg.angular.z = om_output 

            vm.vel_pub.publish(output_msg)
        rate.sleep()
        

if __name__ == '__main__':
    main(sys.argv)
