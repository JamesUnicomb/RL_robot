#!/usr/bin/env python

import rospy
from rl_path_follow import PathFollow

def main():
    rospy.init_node('path_follow_test')

    pf = PathFollow(load_model = False, record_data = False)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()

if __name__=='__main__':
    main()
