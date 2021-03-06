#!/usr/bin/env python

import rospy
from rl_path_follow import PathFollow

import tensorflow as tf

def main():
    rospy.init_node('path_follow_test')

    pf = PathFollow(load_model = True, record_data = False)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pf.set_velocity(pf.nu)
        rate.sleep()

if __name__=='__main__':
    main()
