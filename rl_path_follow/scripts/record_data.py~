#!/usr/bin/env python

import os, time

import rospy, rospkg
from rl_path_follow import PathFollow

import numpy as np

import atexit

import scipy.io as io

def main():
    pf = PathFollow(load_model = False, record_data = False)

    image_data    = []
    velocity_data = []

    rospy.init_node('path_follow_record')

    def save_data():
        training_list = os.listdir(pf.pkg_path + '/data/training_data')
        
        os.mkdir(pf.pkg_path + '/data/training_data/data_set_' + str(len(training_list)))

        io.savemat(pf.pkg_path + '/data/training_data/data_set_' + str(len(training_list)) + '/training_data', 
                                                         {'velocities': np.array(velocity_data), 'images': np.array(image_data)})

        print('shutting down program and saving data')

    atexit.register(save_data)

    rate = rospy.Rate(10)

    print(pf.pkg_path)

    while not rospy.is_shutdown():
        vels, image = pf.return_data()

        if vels[0] > 0.1:
            try:
                image_data.append(np.array(image).reshape(60,80,1))
                velocity_data.append(vels)
            except ValueError:
                pass

        rate.sleep()


if __name__=='__main__':
    main()
