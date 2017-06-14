#!/usr/bin/env python

import rospy
from learner_bot import RLData

rospy.init_node('test_node', anonymous=True)

def main():
    robot_data = RLData()
    
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        print robot_data.laser_array
        rate.sleep()

if __name__=='__main__':
    main()
