#!/usr/bin/env python

import rospy

from random_walker import iRobotRandomWalker

rospy.init_node('random_walker_node', anonymous=True)

def main():
    create_walker = iRobotRandomWalker()
    create_walker.random_walker(15.0)

if __name__=='__main__':
    main()
