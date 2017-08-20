#!/usr/bin/env python

import os, time

import rospy
from rl_path_follow import PathFollow

import tensorflow as tf

import scipy.io as io
import numpy as np

def main():
    pf = PathFollow(load_model = True, record_data = False)

    data = io.loadmat(pf.pkg_path + '/data/training_data/' + os.listdir(pf.pkg_path + '/data/training_data')[0] + '/training_data.mat')
    trX = data['images']
    trY = data['velocities']

    trX = np.array(trX).reshape(-1,1,60,80)
    trY = np.array(trY)[:,1].reshape(-1,1)

    l1 = np.mean(np.abs(pf.prediction(trX) - trY))
    l2 = np.mean(np.square(pf.prediction(trX) - trY))

    print 0.5 * l1 + 0.5 * l2

if __name__=='__main__':
    main()
