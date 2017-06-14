#!/usr/bin/env python

import rospy, rospkg

import numpy as np

from create_env import createAPI
from learner_bot import RLData

from std_msgs.msg import Float64

rospy.init_node('random_policy_learning')

reset_pub = rospy.Publisher('create/reset', Float64, queue_size=2)

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('learner_bot')

def main():
    robot_data = RLData()
    create = createAPI()

    reset_pub.publish(240.0)

    rate = rospy.Rate(10)

    k = 0

    while not rospy.is_shutdown():
        robot_feedback = False
        episode_data = []
        step = 0
        while (not robot_feedback)and(create.in_episode)and(step<60):
            robot_feedback = (create.left_bumper)or(create.right_bumper)
            out_data = [robot_data.lin_vel, robot_data.ang_vel] + list(robot_data.laser_array)
            episode_data.append(out_data)
            rospy.loginfo('len_out: %d, len_list: %d' % (len(out_data), len(episode_data)))
            rate.sleep()
            step += 1
        if (len(episode_data)>1)and(step<55):
            np.save(pkg_path + '/data/rl_test' + str(k), episode_data)
            rospy.loginfo('data_saved!')
            k += 1


if __name__=='__main__':
    main()
